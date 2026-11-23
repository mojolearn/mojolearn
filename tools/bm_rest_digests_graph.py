#!/usr/bin/env python3
"""Output digests of the operations lane bm-rest-graph moved to typed RUNs
(2026-11-23): SAGEConv's host-form forward and backward (one native saved
state) and ProphetForecaster.predict's setup (one typed fitted plan). A
testing tool, never product execution.

One line per configuration, `<name> sha256=<digest of every output's bytes>`
or `<name> error <type>: <first line>` (a refusal is digested by its words).
Run it on two host builds of the CPU column (before and after the change) and
diff the files: equal lines mean equal bits.

    PYTHONPATH=<repo>/python python3 tools/bm_rest_digests_graph.py > out.txt

SAGE configurations: every aggregation (mean, sum, max) x normalize x
root_weight x bias x project (48), each a forward, a backward and the
gradients the backward publishes (lin_l's weight and bias, the root weight,
the projection's weight and bias), plus a second forward + backward on the
cached graph and the backward-before-forward refusal. Prophet: predict of a
default fit, with holidays, multiplicative, a batch of series, a datetime
column, an empty `t`, a missing holiday matrix, a changed seasonality
(schema refusal), each digesting the forecast and the published `trend_`.

Parallel plan: the configurations are independent, one per worker process
(at most the allocated cores, at most the configuration count).
"""
import hashlib
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor


def _graph(n, k, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, size=n * k)
    dst = rng.integers(0, n, size=n * k)
    # a few duplicate edges (ties for max) and an isolated node (n - 1)
    keep = (src != n - 1) & (dst != n - 1)
    return np.stack([src[keep], dst[keep]]).astype(np.int64)


def _sage_cases():
    cases = {}
    for aggr, norm, root, bias, project in itertools.product(("mean", "sum", "max"), (False, True), (True, False),
                                                             (True, False), (False, True)):
        name = f"sage_{aggr}_n{int(norm)}_r{int(root)}_b{int(bias)}_p{int(project)}"
        cases[name] = ("sage", dict(aggr=aggr, normalize=norm, root_weight=root, bias=bias, project=project))
    cases["sage_mean_cached_again"] = ("sage_again", dict(aggr="mean", normalize=True, project=True))
    cases["sage_max_cached_again"] = ("sage_again", dict(aggr="max", root_weight=False))
    cases["sage_backward_first"] = ("sage_backward_first", dict(aggr="mean"))
    return cases


def _prophet_cases():
    return {
        "prophet_default": ("prophet", dict()),
        "prophet_holidays": ("prophet", dict(holidays=True)),
        "prophet_multiplicative": ("prophet", dict(kw=dict(seasonality_mode="multiplicative"))),
        "prophet_batch": ("prophet", dict(batch=True)),
        "prophet_datetime": ("prophet", dict(datetime=True)),
        "prophet_empty_t": ("prophet", dict(empty=True)),
        "prophet_missing_holidays": ("prophet", dict(holidays=True, drop_holidays=True)),
        "prophet_schema_changed": ("prophet", dict(change_schema=True)),
        "prophet_custom_season": ("prophet", dict(custom=True)),
    }


def _cases():
    cases = _sage_cases()
    cases.update(_prophet_cases())
    return cases


def _digest(arrays):
    import numpy as np
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def _sage(kw, again=False):
    import numpy as np
    import mojolearn as ml
    rng = np.random.default_rng(3)
    n, d, f = 40, 6, 5
    x = rng.standard_normal((n, d)).astype(np.float32)
    ei = _graph(n, 3, 4)
    conv = ml.SAGEConv(d, f, random_state=7, **kw)
    outs = []
    for _ in range(2 if again else 1):  # driver-loop: the forward again on the cached graph
        y = conv.forward(x, ei)
        g = rng.standard_normal(y.shape).astype(np.float32)
        dx = conv.backward(g)
        outs += [y, dx, conv.lin_l.grad_weight_, conv.lin_l.grad_bias_]
        if kw.get("root_weight", True):
            outs.append(conv.grad_weight_r_)
        if kw.get("project", False):
            outs += [conv.lin.grad_weight_, conv.lin.grad_bias_]
        x = (x * np.float32(0.5)).astype(np.float32)
    return outs


def _sage_backward_first(kw):
    import numpy as np
    import mojolearn as ml
    conv = ml.SAGEConv(6, 5, random_state=7, **kw)
    conv.backward(np.zeros((4, 5), dtype=np.float32))
    return []


def _prophet(opts):
    import numpy as np
    import mojolearn as ml
    rng = np.random.default_rng(0)
    n = 200
    t = 18000.0 + np.arange(n, dtype=np.float64)
    y = (0.02 * np.arange(n) + np.sin(2 * np.pi * np.arange(n) / 7) + 0.1 * rng.standard_normal(n))
    y = y.astype(np.float32)
    if opts.get("kw", {}).get("seasonality_mode") == "multiplicative":
        y = (y + np.float32(5.0)).astype(np.float32)
    hol = ((np.arange(n + 14) % 50) == 7).astype(np.float32)[:, None]
    model = ml.ProphetForecaster(**opts.get("kw", {}))
    if opts.get("custom"):
        model.add_seasonality("monthly", period=30.5, fourier_order=3)
    yy = np.stack([y, 2 * y]) if opts.get("batch") else y
    tt = t
    tf = 18000.0 + np.arange(n, n + 14, dtype=np.float64)
    if opts.get("datetime"):
        tt = (np.datetime64("2019-04-14") + np.arange(n)).astype("datetime64[ns]")
        tf = (np.datetime64("2019-04-14") + np.arange(n, n + 14)).astype("datetime64[ns]")
    if opts.get("holidays"):
        model.fit(tt, yy, holidays=hol[:n])
    else:
        model.fit(tt, yy)
    if opts.get("change_schema"):
        model.seasonalities["weekly"]["fourier_order"] = 1
    if opts.get("empty"):
        tf = tf[:0]
    if opts.get("holidays") and not opts.get("drop_holidays"):
        yhat = model.predict(tf, holidays=hol[n:n + len(tf)])
    else:
        yhat = model.predict(tf)
    return [yhat, model.trend_, model.params_]


def _run(name):
    kind, kw = _cases()[name]
    try:
        if kind == "sage":
            outs = _sage(kw)
        elif kind == "sage_again":
            outs = _sage(kw, again=True)
        elif kind == "sage_backward_first":
            outs = _sage_backward_first(kw)
        else:
            outs = _prophet(kw)
        return f"{name} sha256={_digest(outs)}"
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def main():
    names = sorted(_cases())
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    workers = max(1, min(int(os.environ.get("BM_REST_WORKERS", cores)), len(names)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_run, names):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
