#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The classical GEMM callers under the ksplit default against the old plan
(docs/lanes/BRIEF_gemm_long_k_2026-09-11.md section 11, job 2).

ENGINEERING_RULES.md section 9: a shared kernel's flip must hold for every
lane it reaches. DEVIATION 2595 made `ksplit` the shipped GEMM plan on NVIDIA
wherever the long-k group rule takes a call through `identical_gemm_into`,
and classical estimators reach that entry. This file times OLS, PCA and GP
on the two section 9 datasets, NYC taxi and Istella-S, once per block under
the trial arm `tuned128` (the old TUNED 128x128 plan) and once under the
shipped default, through the PUBLIC estimators of IDENTICAL bindings built
with `-D MOJOLEARN_GEMM_ARM_TRIAL=1`, so `MOJOLEARN_GEMM_ARM` picks the plan
on every GEMM call. `tools/gemm_ksplit_classical_leg.sh` drives it on the
box.

    python tools/gemm_ksplit_classical_ab.py smoke
    python tools/gemm_ksplit_classical_ab.py time --lane ols --dataset taxi \\
        --arm-name tuned128 --block 1 [--rounds 3] [--rows 4000000] \\
        [--gp-train 4000] [--gp-test 1000] [--prep standardize] [--device NAME]
    python tools/gemm_ksplit_classical_ab.py verdict --out DIR [--lanes gp,ols,pca]

WHY NOT bench/speed/classical_speed_main.mojo. That driver is the classical
lanes' timing entry point, and it times `ols` and `pca` on a splitmix64
generator (4,000,000 x 32) and `gp` on a 12 x 3 correctness fixture; none of
its lanes reads taxi or Istella-S. The only real-data classical harness in
the tree is the kNN gate's (`tools/knn_selection_gate.py`, through the public
estimator of a trial binding). This file follows that pattern and the FSPEED
record format of `classical_speed_main.mojo` and `tools/speed_cuml_arm.py`,
so `tools/flip_verdict.py` reads its logs unchanged.

THE DATA. The trees harness's caches (`tools/speed_gbdt_arm.py --download
taxi|istella`, under GBM_BENCH_DATA), loaded through that harness's own
`load_taxi` / `load_istella` with `regression=True`, so the rows, the split
and the column selection cannot drift from the lane that built them. Taxi
takes its 11 `TAXI_NUMERIC` columns (target: fare), Istella-S its 220
features (target: the 0..4 grade). OLS and PCA take the first `--rows` train
rows (section 9: 4,000,000; Istella-S's train split holds 2,043,304, so it
takes them all); quality is scored on the harness's test rows. GP takes the
first `--gp-train` train rows and the first `--gp-test` test rows; section 9
declares no GP shape, so 4,000 is a rung of the declared GP ladder
(`bench/speed/classical_ladder_main.mojo`) where the cubic factor, not the
launch, is the cost.

`--prep standardize` (the default) centers and scales every column with the
training rows' float64 mean and standard deviation (a zero deviation stays
1) and applies the same map to the test rows. It is the same bytes for both
arms, and it is needed: Istella-S carries float32-max sentinels, and a
float32 Gram of raw sentinels overflows to inf. GP also standardizes `y` and
scores RMSE in the original units. `--prep raw` feeds the cache as is.

THE RECORD (one process = one lane, one dataset, one arm, one block):

    FSPEED-HEADER family=classical lane=<l> arm=ours mode=IDENTICAL device=<d> rounds=<n> size=section9
    FSPEED-NOTE lane=<l> arm=ours gemm_arm=<tuned128|default> gemm_plan=<label> block=<b> ...
    FSPEED-GEMM lane=<l> caller=<name.dataset> op=<NN|NT|TN> m=<m> n=<n> k=<k>
    FSPEED-WARMUP lane=<l> arm=ours shape=<tag> ms=<float>
    FSPEED lane=<l> arm=ours shape=<tag> round=<i> ms=<float> hash=<16 hex>
    FSPEED-ACC lane=<l> arm=ours metric=<rmse|mse> value=<float>

`arm=ours` on both sides is deliberate: flip_verdict compares its BEFORE logs
(the `tuned128` blocks) against its AFTER logs (the default blocks) at one
lane and one arm name. `hash` is FNV-1a64 over the output bytes (OLS coef_
and intercept, PCA components_ and explained_variance_, GP predictive means).
IDENTICAL promises the same bits under both plans (brief section 5), so every
hash of a caller and dataset must agree across arms and blocks; the verdict
calls a disagreement an IDENTITY-BREAK. `FSPEED-GEMM` names each GEMM the
caller issues through `identical_gemm_into` at this shape; the leg turns
each into a DISPATCH line with the Mojo dispatch itself
(`bench/gemm_step_price_main.mojo` label mode), which says whether the ksplit
default takes it. No rule is re-spelled here.

THE VERDICT, per caller: flip_verdict's own output (both ratios, the geomean,
the quality lines, FLIP or NO FLIP, where FLIP means the default is faster),
then one line of the regression rule this A/B exists for. A caller REGRESSES
when the geometric mean of its two default/tuned128 median ratios exceeds
1 plus the measured noise (the largest block-to-block spread of either arm's
block medians on either dataset), or when flip_verdict marks a quality
metric WORSE. It HOLDS otherwise. A dataset missing on either side is
UNMEASURED; one block per arm leaves the noise unmeasured and the band 0.
"""
import argparse
import contextlib
import glob
import hashlib
import importlib.util
import io
import math
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LANES = ("gp", "ols", "pca")
DATASETS = ("taxi", "istella")
ARM_NAMES = ("tuned128", "default")
#: `cholesky/checks/potrf.mojo::CHOL_NB_PINNED`, the IDENTICAL panel width and
#: so `k` of the Cholesky trailing update's GEMM. Transcribed for the
#: FSPEED-GEMM record only; whether the default takes that call is the Mojo
#: dispatch's answer on the box (dispatch.txt), never this file's.
CHOL_NB_PINNED = 32
PCA_COMPONENTS = 8
GP_ALPHA = 0.1

EXIT_OK, EXIT_VERDICT, EXIT_REFUSED = 0, 1, 3


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fnv1a64(data):
    """FNV-1a64, byte at a time, as `core/identity_trace.mojo::fnv1a64_bytes`."""
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def _say(line):
    print(line, flush=True)


def _no_spaces(text):
    return re.sub(r"\s+", "_", str(text)).strip("_") or "-"


def _import_mojolearn():
    os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
    python_dir = os.path.join(ROOT, "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    import mojolearn  # noqa: PLC0415
    return mojolearn


# ------------------------------------------------------------------- data

def load_dataset(dataset, rows):
    """(x_train, y_train, x_test, y_test) as C-contiguous float32."""
    import numpy as np  # noqa: PLC0415
    harness = _load_module("speed_gbdt_arm", os.path.join(HERE, "speed_gbdt_arm.py"))
    if dataset == "taxi":
        data = harness.load_taxi("shipped", rows_cap=rows, regression=True)
        cols = [list(harness.TAXI_FEATURES).index(c) for c in harness.TAXI_NUMERIC]
        x_train = data.X_train[:, cols]
        x_test = data.X_test[:, cols]
    elif dataset == "istella":
        data = harness.load_istella("shipped", rows_cap=rows, regression=True)
        x_train, x_test = data.X_train, data.X_test
    else:
        raise ValueError("unknown dataset %r; choose from %s" % (dataset, DATASETS))

    def f32(a):
        return np.ascontiguousarray(a, dtype=np.float32)

    return f32(x_train), f32(data.y_train), f32(x_test), f32(data.y_test)


def standardize(x_train, x_test):
    """Training-row float64 column mean and deviation, applied to both."""
    import numpy as np  # noqa: PLC0415
    x64 = x_train.astype(np.float64)
    mean = x64.mean(axis=0)
    dev = x64.std(axis=0)
    dev[~(dev > 0.0)] = 1.0
    out_train = np.ascontiguousarray(((x64 - mean) / dev).astype(np.float32))
    del x64
    out_test = np.ascontiguousarray(((x_test.astype(np.float64) - mean) / dev).astype(np.float32))
    return out_train, out_test


# ------------------------------------------------------------------ lanes

def _f32(np, a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def lane_ols(mojolearn, np, x, y, xq, yq, dataset):
    from mojolearn.linear_model import LinearRegression  # noqa: PLC0415
    rows, d = x.shape

    def fit():
        return LinearRegression().fit(x, y)

    def digest(model):
        return fnv1a64(_f32(np, model.coef_).tobytes()
                       + np.float32(model.intercept_).tobytes())

    def quality(model):
        pred = np.asarray(model.predict(xq), dtype=np.float64)
        return "rmse", float(np.sqrt(np.mean((pred - yq.astype(np.float64)) ** 2)))

    # glm/estimator.mojo::ols_fit_host -> lstsq_eig -> core/gemm.mojo::gemm_tn,
    # whose IDENTICAL arm on NVIDIA and AMD is gemm_tn_identical_v1 ->
    # identical_gemm_into(m = n = d, k = rows, OP_TN). The inverse's gemm_nt
    # is the pinned NT kernel, not the entry.
    shapes = [("ols.gram", "TN", d, d, rows)]
    return fit, digest, quality, "%s-%dx%d" % (dataset, rows, d), shapes


def lane_pca(mojolearn, np, x, y, xq, yq, dataset):
    from mojolearn.decomposition import PCA  # noqa: PLC0415
    rows, d = x.shape

    def fit():
        return PCA(n_components=PCA_COMPONENTS, svd_solver="covariance_eigh").fit(x)

    def digest(model):
        return fnv1a64(_f32(np, model.components_).tobytes()
                       + _f32(np, model.explained_variance_).tobytes())

    def quality(model):
        back = np.asarray(model.inverse_transform(model.transform(xq)), dtype=np.float64)
        diff = back - xq.astype(np.float64)
        return "mse", float(np.mean(diff * diff))

    # decomposition/estimator.mojo::pca_fit_host -> pca_fit -> gemm_tn, the
    # same OP_TN Gram as OLS (m = n = d, k = rows).
    shapes = [("pca.cov", "TN", d, d, rows)]
    return (fit, digest, quality,
            "%s-%dx%dc%d" % (dataset, rows, d, PCA_COMPONENTS), shapes)


def lane_gp(mojolearn, np, x, y, xq, yq, dataset):
    from mojolearn import RBF, GaussianProcessRegressor  # noqa: PLC0415
    n, d = x.shape
    n_star = xq.shape[0]
    y64 = y.astype(np.float64)
    y_mean = float(y64.mean())
    y_dev = float(y64.std()) or 1.0
    yz = np.ascontiguousarray(((y64 - y_mean) / y_dev).astype(np.float32))

    def fit():
        model = GaussianProcessRegressor(kernel=RBF(length_scale=float(math.sqrt(d))),
                                         alpha=GP_ALPHA)
        model.fit(x, yz)
        info = getattr(model, "info_", 0)
        if info:
            raise RuntimeError("the kernel matrix did not factor: info_=%r" % (info,))
        return model, model.predict(xq)

    def digest(result):
        return fnv1a64(_f32(np, result[1]).tobytes())

    def quality(result):
        pred = np.asarray(result[1], dtype=np.float64) * y_dev + y_mean
        return "rmse", float(np.sqrt(np.mean((pred - yq.astype(np.float64)) ** 2)))

    # gaussian_process/estimator.mojo: the posterior mean is
    # identical_gemm_into(n_star, 1, n_train, OP_TN); the Cholesky
    # (cholesky/checks/potrf.mojo) issues identical_gemm_into(n_trail, n_trail,
    # w, OP_NT) per panel, largest at n_trail = n - w. The Gram and the cross
    # covariance are elementwise kernels, not GEMM.
    trail = max(n - CHOL_NB_PINNED, 0)
    shapes = [("gp.mean", "TN", n_star, 1, n),
              ("gp.chol_trailing", "NT", trail, trail, CHOL_NB_PINNED)]
    return fit, digest, quality, "%s-%dx%ds%d" % (dataset, n, d, n_star), shapes


LANE_BUILDERS = {"ols": lane_ols, "pca": lane_pca, "gp": lane_gp}


# ------------------------------------------------------------------- time

def cmd_time(args):
    import numpy as np  # noqa: PLC0415
    lane, dataset = args.lane, args.dataset
    gemm_arm = os.environ.get("MOJOLEARN_GEMM_ARM", "")
    want_env = "tuned128" if args.arm_name == "tuned128" else ""
    if gemm_arm != want_env:
        _say("FSPEED-REFUSED lane=%s arm=ours reason=--arm-name=%s_but_MOJOLEARN_GEMM_ARM=%s"
             % (lane, args.arm_name, gemm_arm or "(unset)"))
        return EXIT_REFUSED
    _say("FSPEED-HEADER family=classical lane=%s arm=ours mode=IDENTICAL device=%s rounds=%d "
         "size=section9" % (lane, _no_spaces(args.device), args.rounds))
    try:
        mojolearn = _import_mojolearn()
        x, y, xq, yq = load_dataset(dataset, args.gp_train if lane == "gp" else args.rows)
        if lane == "gp":
            x, y = x[:args.gp_train], y[:args.gp_train]
            xq, yq = xq[:args.gp_test], yq[:args.gp_test]
        if args.prep == "standardize":
            x, xq = standardize(x, xq)
        x = np.ascontiguousarray(x, dtype=np.float32)
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        fit, digest, quality, shape, shapes = LANE_BUILDERS[lane](
            mojolearn, np, x, y, xq, yq, dataset)
    except Exception as exc:  # noqa: BLE001
        _say("FSPEED-REFUSED lane=%s arm=ours reason=%s" % (lane, _no_spaces(exc)))
        return EXIT_REFUSED
    _say("FSPEED-NOTE lane=%s arm=ours gemm_arm=%s gemm_plan=%s block=%s dataset=%s prep=%s "
         "rows=%d features=%d test_rows=%d x_sha256=%s mojolearn=%s"
         % (lane, args.arm_name, _no_spaces(os.environ.get("MOJOLEARN_GEMM_PLAN_LABEL", "unlabeled")),
            args.block, dataset, args.prep, x.shape[0], x.shape[1], xq.shape[0],
            hashlib.sha256(x.tobytes()).hexdigest()[:16],
            _no_spaces(getattr(mojolearn, "__file__", "?"))))
    for caller, op, m, n, k in shapes:
        _say("FSPEED-GEMM lane=%s caller=%s.%s op=%s m=%d n=%d k=%d"
             % (lane, caller, dataset, op, m, n, k))
    hashes = []
    last = None
    for r in range(args.rounds + 1):
        try:
            t0 = time.perf_counter_ns()
            result = fit()
            t1 = time.perf_counter_ns()
            h = digest(result)
        except Exception as exc:  # noqa: BLE001
            _say("FSPEED-REFUSED lane=%s arm=ours reason=%s" % (lane, _no_spaces(exc)))
            return EXIT_REFUSED
        ms = (t1 - t0) / 1.0e6
        if r == 0:
            _say("FSPEED-WARMUP lane=%s arm=ours shape=%s ms=%.3f" % (lane, shape, ms))
            mode = getattr(result[0] if isinstance(result, tuple) else result,
                           "numeric_mode_used", None)
            if callable(mode):
                _say("FSPEED-NOTE lane=%s arm=ours numeric_mode_used=%s" % (lane, _no_spaces(mode())))
        else:
            _say("FSPEED lane=%s arm=ours shape=%s round=%d ms=%.3f hash=%s"
                 % (lane, shape, r, ms, h))
            hashes.append(h)
        last = result
    if len(set(hashes)) > 1:
        _say("FSPEED-NOTE lane=%s arm=ours hash moved across rounds: %s"
             % (lane, " ".join(sorted(set(hashes)))))
    try:
        metric, value = quality(last)
    except Exception as exc:  # noqa: BLE001
        _say("FSPEED-NOTE lane=%s arm=ours quality not computed: %s" % (lane, _no_spaces(exc)))
        return EXIT_OK
    if math.isfinite(value):
        _say("FSPEED-ACC lane=%s arm=ours metric=%s value=%.9g" % (lane, metric, value))
    else:
        # flip_verdict refuses a non-finite before value; the round hashes
        # still carry the equality of the outputs.
        _say("FSPEED-NOTE lane=%s arm=ours quality %s=%r is not finite, no FSPEED-ACC line"
             % (lane, metric, value))
    return EXIT_OK


# ------------------------------------------------------------------ smoke

def cmd_smoke(args):
    """Import the IDENTICAL package and fit each estimator once on a tiny
    generated table, so a missing or unloadable binding is named before any
    dataset is read. A check, never a timing."""
    import numpy as np  # noqa: PLC0415
    mojolearn = _import_mojolearn()
    from mojolearn import RBF, GaussianProcessRegressor  # noqa: PLC0415
    from mojolearn.decomposition import PCA  # noqa: PLC0415
    from mojolearn.linear_model import LinearRegression  # noqa: PLC0415
    gen = np.random.default_rng(20260911)
    x = np.ascontiguousarray(gen.standard_normal((64, 4)), dtype=np.float32)
    y = np.ascontiguousarray(x @ np.arange(1, 5, dtype=np.float32), dtype=np.float32)
    print("mojolearn=%s mode_env=%s gemm_arm=%s" % (
        mojolearn.__file__, os.environ.get("MOJOLEARN_NUMERIC_MODE"),
        os.environ.get("MOJOLEARN_GEMM_ARM", "")), flush=True)
    ols = LinearRegression().fit(x, y)
    print("ols coef=%s mode=%s" % (np.asarray(ols.coef_).tolist(), ols.numeric_mode_used()), flush=True)
    pca = PCA(n_components=2, svd_solver="covariance_eigh").fit(x)
    print("pca explained_variance=%s" % np.asarray(pca.explained_variance_).tolist(), flush=True)
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=2.0), alpha=GP_ALPHA).fit(x[:16], y[:16])
    print("gp info=%r mean0=%r" % (getattr(gp, "info_", None),
                                   float(np.asarray(gp.predict(x[:2]))[0])), flush=True)
    return EXIT_OK


# ---------------------------------------------------------------- verdict

def _block_logs(out, lane, dataset, arm):
    return sorted(glob.glob(os.path.join(out, "%s.%s.%s.*.log" % (lane, dataset, arm))))


def _fmt(x, digits=4):
    return "-" if x is None else "%.*f" % (digits, x)


def cmd_verdict(args):
    fv = _load_module("flip_verdict", os.path.join(HERE, "flip_verdict.py"))
    lanes = [l for l in args.lanes.split(",") if l]
    print("RULE a caller REGRESSES when geomean(default/tuned128 median ratio over taxi and "
          "Istella-S) > 1 + noise (the largest block-to-block spread of either arm's block "
          "medians on either dataset), or when a quality metric is WORSE; else it HOLDS. "
          "Every output hash must agree across arms and blocks (else IDENTITY-BREAK).",
          flush=True)
    summary = []
    failed = False
    for lane in lanes:
        print("== caller %s ==" % lane, flush=True)
        logs = {}
        argv = ["--lane", lane, "--arm", "ours"]
        for ds in DATASETS:
            for arm in ARM_NAMES:
                logs[(ds, arm)] = _block_logs(args.out, lane, ds, arm)
            if logs[(ds, "tuned128")]:
                argv += ["--%s-before" % ds] + logs[(ds, "tuned128")]
            if logs[(ds, "default")]:
                argv += ["--%s-after" % ds] + logs[(ds, "default")]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = fv.main(argv)
            except SystemExit as exc:
                code = exc.code
        text = out.getvalue()
        sys.stdout.write(text)
        sys.stdout.write(err.getvalue())
        lines = text.strip().splitlines()
        fv_last = lines[-1] if lines else "no output, exit %r" % (code,)
        worse = [ds for ds in DATASETS
                 if re.search(r"^quality %s .* WORSE$" % ds, text, re.M)]

        ratios, noises, witness = {}, {}, {}
        for ds in DATASETS:
            meds, every, hashes = {}, {}, set()
            for arm in ARM_NAMES:
                meds[arm], every[arm] = [], []
                for path in logs[(ds, arm)]:
                    try:
                        rounds, _accs = fv.parse_files([path])
                    except fv.FlipError as exc:
                        print("verdict %s %s: %s" % (lane, ds, exc), flush=True)
                        continue
                    mine = [r for r in rounds if r.lane == lane and r.arm == "ours"]
                    hashes.update(r.hash for r in mine)
                    if mine:
                        ms = [r.ms for r in mine]
                        meds[arm].append(statistics.median(ms))
                        every[arm].extend(ms)
            if not every["tuned128"] or not every["default"]:
                print("time %s %s: unmeasured (tuned128 rounds %d, default rounds %d)"
                      % (lane, ds, len(every["tuned128"]), len(every["default"])), flush=True)
                continue
            base = statistics.median(every["tuned128"])
            if base <= 0.0:
                print("time %s %s: tuned128 median is not positive" % (lane, ds), flush=True)
                continue
            ratios[ds] = statistics.median(every["default"]) / base
            spreads = [max(v) / min(v) - 1.0 for v in meds.values() if len(v) >= 2 and min(v) > 0.0]
            noises[ds] = max(spreads) if spreads else None
            witness[ds] = "equal" if len(hashes) == 1 and "-" not in hashes else "MOVED"
            beyond = noises[ds] is not None and ratios[ds] > 1.0 + noises[ds]
            print("time %s %s: default/tuned128=%.4f block_noise=%s blocks=%d/%d witness=%s%s"
                  % (lane, ds, ratios[ds], _fmt(noises[ds]), len(meds["tuned128"]),
                     len(meds["default"]), witness[ds],
                     " (slower beyond noise on this dataset)" if beyond else ""), flush=True)

        geomean = noise = None
        if all(ds in ratios for ds in DATASETS):
            geomean = math.sqrt(ratios["taxi"] * ratios["istella"])
            if all(noises[ds] is not None for ds in DATASETS):
                noise = max(noises[ds] for ds in DATASETS)
            band = noise if noise is not None else 0.0
            if any(witness[ds] != "equal" for ds in DATASETS):
                verdict, reason = "IDENTITY-BREAK", "hash"
            elif worse:
                verdict, reason = "REGRESSES", "quality:" + ",".join(worse)
            elif geomean > 1.0 + band:
                verdict, reason = "REGRESSES", "time"
            else:
                verdict, reason = "HOLDS", "-"
        else:
            verdict, reason = "UNMEASURED", "missing:" + ",".join(
                ds for ds in DATASETS if ds not in ratios)
        failed = failed or verdict != "HOLDS"
        line = ("caller=%s verdict=%s reason=%s geomean=%s noise=%s taxi=%s istella=%s "
                "witness_taxi=%s witness_istella=%s flip_verdict=[%s]"
                % (lane, verdict, reason, _fmt(geomean), _fmt(noise) if noise is not None
                   else "unmeasured", _fmt(ratios.get("taxi")), _fmt(ratios.get("istella")),
                   witness.get("taxi", "-"), witness.get("istella", "-"), fv_last))
        print(line, flush=True)
        summary.append("%s=%s" % (lane, verdict))
    print("CLASSICAL KSPLIT A/B " + " ".join(summary), flush=True)
    return EXIT_VERDICT if failed else EXIT_OK


# ------------------------------------------------------------------- main

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("time", help="time one lane on one dataset under one arm")
    t.add_argument("--lane", required=True, choices=LANES)
    t.add_argument("--dataset", required=True, choices=DATASETS)
    t.add_argument("--arm-name", required=True, choices=ARM_NAMES)
    t.add_argument("--block", required=True)
    t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--rows", type=int, default=4_000_000)
    t.add_argument("--gp-train", type=int, default=4000)
    t.add_argument("--gp-test", type=int, default=1000)
    t.add_argument("--prep", choices=("standardize", "raw"), default="standardize")
    t.add_argument("--device", default="unknown")
    sub.add_parser("smoke", help="import the IDENTICAL package and fit each estimator once, tiny")
    v = sub.add_parser("verdict", help="per-caller verdict from the logs in --out")
    v.add_argument("--out", required=True)
    v.add_argument("--lanes", default=",".join(LANES))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "time":
        if args.rounds < 1:
            raise SystemExit("--rounds must be at least 1")
        return cmd_time(args)
    if args.cmd == "smoke":
        return cmd_smoke(args)
    return cmd_verdict(args)


if __name__ == "__main__":
    sys.exit(main())
