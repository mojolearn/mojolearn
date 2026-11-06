#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ONE-GPU MULTI-DEVICE GATE (Andrew 2026-11-06): logical devices on one card, same bits as one device.

For each lane in LANES below (a short, fixed, hand-written list; adding a lane = adding a line), this tool calls
the PUBLIC multi-device entry a user would call (`mojolearn.parallel_classical.fit_svm(..., devices=...)` and its
siblings) three times on the INSTALLED wheel: with `devices=(0,)` (the single-device plan), `(0, 0)` and
`(0, 0, 0)`. The repeated index is admitted only by the test-only option MOJOLEARN_DEVICE_PLAN_LOGICAL=1
(core/device_plan.mojo `device_plan_logical`); nothing else in the route reads it, so the 2- and 3-device arms run
the real split, the real per-rank owners (each rank opens its own `DeviceContext(device_id=0)`, i.e. its own
stream/queue on the one card), the real peer_clone / enqueue_copy_to / transfer_bytes moves between ranks (no
dispatcher branches on "same device": they branch on the rank) and the real fold. The output digests of the three
arms must be equal. One run per arm (identity runs once; a mismatch is a bug to fix, never a reason to rerun).

It prints one line per lane

    SHARDCHECK <lane> single=<d> n2=<d> n3=<d> MATCH|DIFFER|FAIL <reason>

writes <out>/shardcheck.json and exits 1 unless every lane on the list is MATCH. FAIL CLOSED: a lane that cannot
run, raises, times out or returns no digest is a FAIL with a one-line reason, never a skip.

SPREAD (lane perf3-cluster-gp-gate, 2026-11-06): equal digests prove nothing when a small fixture keeps every
shard on rank 0 (an owner loop's `active = min(count, tiles)` is 1, or an entry routes a small input to its
one-device path). Each arm therefore records how many ranks received work: the number of distinct ranks whose
device an owner loop opened for its shard under the bound plan (core/device_plan.mojo `used`, set by
`DevicePlan.device` / `device_plan_device` exactly where each owner loop opens a rank's DeviceContext), read
through every loaded binding's `device_plan_log()` -> (plans, ranks). A 2- or 3-device arm whose work reached
fewer than 2 ranks is a FAIL ("spread"), and so is a binding that does not report ranks (built before the
counter). The fixtures are sized from the dispatchers' split rules (see THE LIST) so that every logical device
of the largest arm receives a shard.

UNCONDITIONAL: the release's one verification round runs it every time, on the NVIDIA and AMD boxes, against the
exact wheel bytes (tools/ops/release_gate_job.sh phase `shardcheck`; tools/release_identity_wheels.py
`single_gate_ok` refuses a box job without a passing shardcheck.json). There is no change detection, no selector,
no closure or identity-table planner here, by design: the list is the list.

The digest is the multi-device verifier's (`mojolearn._verify_distributed.digest`: sha256 of the raw C-order bytes
with shape and dtype); an arm's short digest is sha256 over its parts' (name, shape, dtype, sha256) lines.

WHAT THIS DOES NOT COVER (say it plainly):
  * real inter-card transfer: every move is on one card (on AMD the cross-rank copies still take the host-staged
    path transfer_bytes uses for cross_device=True, so that code runs, but no peer link is crossed);
  * two physical devices or two device runtimes: every rank's context is on the same card;
  * two card models (mixed hardware in one plan);
  * Apple: the door refuses any device list but (0,) on Metal (bindings/parallel_plan_door.mojo `plan_devices`),
    multi-device is not a Metal feature, and this option does not change that; the CPU column: a CPU-only install
    has no cooperative multi-device plan (the same door refuses it), so there is nothing to check there;
  * the families not on the list (QR panels, hierarchy and neighbors kNN rows, GP covariance rows, the optimizer /
    training and autograd distributed programs, which open DeviceContext(device_id=rank) without a plan).
tools/two_device_checks.sh remains THE real two-GPU check (2x L40S; run it after any multi_gpu.mojo or device plan
change).

Parallel: lanes are independent processes, run at once up to --jobs (default: every lane, capped at the host's
cores). They share the one GPU, which serialises their kernels; each process holds its own runtime context
(a few hundred MB of device memory), which bounds --jobs on a small card, not the host's cores.

Usage (on the box, with the venv the exact wheels were installed into):
    python3 tools/logical_shard_identity.py --python <venv>/bin/python --vendor nvidia|amd --out DIR [--jobs N]
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCHEMA = "mojolearn.logical-shard.v1"
SUMMARY = "shardcheck.json"
LOGICAL_ENV = "MOJOLEARN_DEVICE_PLAN_LOGICAL"
#: the device counts every lane runs: 1 is the single-device plan, the rest are logical devices on one card
COUNTS = (1, 2, 3)
LANE_TIMEOUT = 1200

# ---------------------------------------------------------------------------------------------------- THE LIST
# lane, dispatcher family (the multi_gpu file whose shard path the call reaches), recipe, fixture, rows, features,
# seed. One line per lane. Every lane runs COUNTS. Recipes are below; each calls only the public API.
#
# FIXTURE SIZES come from each dispatcher's split rule (read 2026-11-06, lane perf3-cluster-gp-gate), not from any
# benchmark shape: every logical device of the largest arm (max(COUNTS) = 3) must receive a shard, with margin.
#   svm-rbf      rows of the kernel matrix, active = min(count, m) (svm/multi_gpu.mojo:121): m >= 3 rows
#   logistic     feature columns of X^T dZ, active = min(count, D) (glm/impl/qn/multi_gpu.mojo:70); binary y only
#                (a C > 1 target takes the tiled multiclass rule, qn_tiled_rule.mojo:53): D >= 3
#   lasso-cd     row leaves of 128 (bench/gemm_shapes.mojo contract_leaf_size), k <= 128 stays single
#                (solver/multi_gpu.mojo:63): rows >= 2*128 + 1; only the column mean / norm passes before the
#                Gram sweeps reach it
#   ridge-gram   row chunks, active = min(count, min(128, ceil(k/kc))) (core/gram_multi_gpu.mojo:262), BUT the
#                IDENTICAL Ridge fit takes linear_gram_fit first (bindings/classical_ridge.mojo:103-109), which
#                has no DevicePlan: OWED, see the reply; the spread check fails this lane until a reachable recipe
#   kmeans       row tiles of mblk = 4*8 = 32 rows below 32 features (cluster/multi_gpu.mojo:84-85,
#                fused_distance_nn/simt_kernel.mojo:201-230): rows >= 2*32 + 1
#   dbscan       query rows, active = min(count, rows) (dbscan/impl/multi_gpu.mojo:47): rows >= 3; the default
#                algorithm='rbc' on NVIDIA/AMD identical takes the bitmap adjacency passes with no plan
#                (dbscan/impl/runner.mojo:780-845), so the recipe uses algorithm='brute'
#   gmm          E-step rows, active = min(count, n) (mixture/multi_gpu.mojo:233): rows >= 3
#   bootstrap    resamples, active = min(count, n_resamples) (resample/multi_gpu.mojo:79): n_resamples >= 3
#   cholesky     trailing-update rows after each 32-wide panel (cholesky/multi_gpu.mojo:89, potrf.mojo:398):
#                n >= 32 + 3; solve over right-hand-side columns, active = min(count, nrhs): nrhs >= 3 (B has 4)
#   iforest      trees, active = min(count, n_estimators) (isolation_forest/multi_gpu.mojo:112): default 100
#   gbdt         one-byte packed feature groups of 4 (border_count=128; pointwise_multi_gpu.mojo:130-132,452):
#                features >= 2*4 + 1
#   forest       groves = min(32, trees), active = min(count, groves) (core/forest_inference_pool.mojo:418):
#                n_estimators >= 3 (the recipe fits 16)
LANES = (
    ("svm-rbf", "svm/multi_gpu.mojo", "svm", "classify", 600, 16, 11),
    ("logistic", "glm/impl/qn/multi_gpu.mojo", "logistic", "classify", 2000, 16, 12),
    ("lasso-cd", "solver/multi_gpu.mojo", "lasso", "regress", 2000, 32, 13),
    ("ridge-gram", "core/gram_multi_gpu.mojo", "ridge", "regress", 4000, 32, 14),
    ("kmeans", "cluster/multi_gpu.mojo", "kmeans", "blobs", 4000, 8, 15),
    ("dbscan", "dbscan/impl/multi_gpu.mojo", "dbscan", "blobs", 1500, 4, 16),
    ("gmm", "mixture/multi_gpu.mojo", "gmm", "blobs", 3000, 4, 17),
    ("bootstrap", "resample/multi_gpu.mojo", "bootstrap", "sample", 1000, 1, 18),
    ("cholesky", "cholesky/multi_gpu.mojo", "cholesky", "spd", 384, 384, 19),
    ("iforest", "isolation_forest/multi_gpu.mojo", "iforest", "regress", 2000, 8, 20),
    ("gbdt", "gbdt/methods/pointwise_multi_gpu.mojo", "gbdt", "regress", 2000, 16, 21),
    ("forest-predict", "core/forest_inference_pool.mojo", "forest", "classify", 2000, 16, 22),
)


def lane_names():
    return [row[0] for row in LANES]


def lane_errors(lanes=LANES):
    """Why the fixed list is malformed, or [] (checked before anything runs, and by the unit tests)."""
    errors = []
    names = [row[0] for row in lanes]
    if not 6 <= len(lanes) <= 12:
        errors.append("the list holds %d lanes, not 6..12" % len(lanes))
    if len(set(names)) != len(names):
        errors.append("duplicate lane names")
    if len({row[1] for row in lanes}) != len(lanes):
        errors.append("two lanes name one dispatcher family (one lane per family)")
    for row in lanes:
        if len(row) != 7:
            errors.append("%s: a lane line has 7 fields" % (row[0],))
            continue
        name, _family, recipe, fixture, rows, features, seed = row
        if recipe not in RECIPES:
            errors.append("%s: unknown recipe %s" % (name, recipe))
        if fixture not in FIXTURES:
            errors.append("%s: unknown fixture %s" % (name, fixture))
        if not all(type(v) is int and v > 0 for v in (rows, features, seed)):
            errors.append("%s: rows, features and seed are positive ints" % name)
    if COUNTS[0] != 1 or len(COUNTS) < 2 or len(set(COUNTS)) != len(COUNTS):
        errors.append("COUNTS must start with the single-device plan (1) and hold at least one logical count")
    return errors


def devices_for(count, device=0):
    """The device tuple of one arm: (device,) for the single-device plan, else `device` repeated."""
    return tuple([device] * count)


# ---------------------------------------------------------------------------------------------------- VERDICTS
def arm_digest(parts):
    """The short digest of one arm: sha256 over its parts' (name, shape, dtype, sha256) lines, or None without
    parts (a lane that produced nothing never matches)."""
    if not parts:
        return None
    text = "\n".join("%s %s %s %s" % (p["name"], p["shape"], p["dtype"], p["sha256"]) for p in parts)
    return hashlib.sha256(text.encode()).hexdigest()


#: the fewest ranks a multi-device arm must spread its work over (the gate's point: at least one real split)
MIN_SPREAD = 2


def lane_verdict(name, doc):
    """(verdict, reason, {count: digest}) of one lane's worker document; FAIL on anything missing, and FAIL when a
    multi-device arm's work reached fewer than MIN_SPREAD ranks (a trivial MATCH)."""
    digests = {}
    if not isinstance(doc, dict) or doc.get("lane") != name:
        return "FAIL", "no worker result", digests
    if doc.get("error"):
        return "FAIL", one_line(doc["error"]), digests
    arms = doc.get("arms") or {}
    for count in COUNTS:
        arm = arms.get(str(count))
        if not isinstance(arm, dict):
            return "FAIL", "arm n%d did not run" % count, digests
        if arm.get("error"):
            return "FAIL", "arm n%d: %s" % (count, one_line(arm["error"])), digests
        d = arm_digest(arm.get("parts"))
        if d is None:
            return "FAIL", "arm n%d produced no digest" % count, digests
        digests[count] = d
    single = digests[COUNTS[0]]
    differ = [c for c in COUNTS[1:] if digests[c] != single]
    if differ:
        return "DIFFER", "n%s differ from the single-device plan" % ",".join(map(str, differ)), digests
    for count in COUNTS[1:]:
        ranks = arms[str(count)].get("ranks")
        if type(ranks) is not int:
            return "FAIL", "arm n%d: no rank record (a binding without the device-plan work counter)" % count, digests
        if ranks < min(MIN_SPREAD, count):
            return ("FAIL", "arm n%d spread its work over %d rank(s), not >= %d: the fixture stays under the "
                    "dispatcher's split" % (count, ranks, min(MIN_SPREAD, count)), digests)
        if ranks > count:
            return "FAIL", "arm n%d reports %d ranks, more than its %d devices" % (count, ranks, count), digests
    return "MATCH", "", digests


def ranks_take(modules):
    """The ranks that received work since the last take, over every loaded binding (`device_plan_log()` ->
    (plans, ranks)); the record is cleared. Where the plan table is one process-wide named global the first
    binding's take clears it for the rest, so the maximum is the answer either way. None when no loaded binding
    reports ranks."""
    best = None
    seen = set()
    for module in list(modules.values()):
        if id(module) in seen:
            continue
        seen.add(id(module))
        try:
            log = getattr(module, "device_plan_log", None)
        except Exception:
            continue
        if not callable(log):
            continue
        got = log()
        if isinstance(got, tuple) and len(got) == 2 and type(got[1]) is int:
            best = got[1] if best is None else max(best, got[1])
    return best


def one_line(text):
    text = " ".join(str(text).split())
    return text[:200]


def shardcheck_line(name, verdict, reason, digests):
    def short(c):
        return (digests.get(c) or "-")[:12]
    arms = " ".join(("single" if c == 1 else "n%d" % c) + "=" + short(c) for c in COUNTS)
    return ("SHARDCHECK %s %s %s %s" % (name, arms, verdict, reason)).rstrip()


def summary_errors(doc):
    """Why a shardcheck.json does not prove every lane on the list matched, or [] (the release's verdict)."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return ["no logical-shard summary"]
    errors = []
    lanes = doc.get("lanes") or {}
    if sorted(lanes) != sorted(lane_names()):
        errors.append("lanes %s, not the list %s" % (sorted(lanes), sorted(lane_names())))
    if list(doc.get("counts") or []) != list(COUNTS):
        errors.append("counts %s, not %s" % (doc.get("counts"), list(COUNTS)))
    for name, row in sorted(lanes.items()):
        if (row or {}).get("verdict") != "MATCH":
            errors.append("%s %s %s" % (name, (row or {}).get("verdict"), (row or {}).get("reason", "")))
    if doc.get("ok") is not True and not errors:
        errors.append("the summary is not ok")
    return errors


# ---------------------------------------------------------------------------------------------------- FIXTURES
# Test control only: the fixtures are built with NumPy in the harness (never inside the library's runtime).
def _fixture_classify(np, rng, rows, features):
    X = rng.standard_normal((rows, features)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int32)
    return X, y


def _fixture_regress(np, rng, rows, features):
    X = rng.standard_normal((rows, features)).astype(np.float32)
    w = rng.standard_normal(features).astype(np.float32)
    y = (X @ w + 0.1 * rng.standard_normal(rows)).astype(np.float32)
    return X, y


def _fixture_blobs(np, rng, rows, features):
    centers = 4.0 * rng.standard_normal((4, features))
    label = rng.integers(0, 4, rows)
    X = (centers[label] + 0.3 * rng.standard_normal((rows, features))).astype(np.float32)
    return X, label.astype(np.int32)


def _fixture_spd(np, rng, rows, features):
    M = rng.standard_normal((rows, rows))
    A = (M @ M.T / rows + np.eye(rows)).astype(np.float32)
    B = rng.standard_normal((rows, 4)).astype(np.float32)
    return A, B


def _fixture_sample(np, rng, rows, features):
    return rng.standard_normal(rows).astype(np.float32), None


FIXTURES = dict(classify=_fixture_classify, regress=_fixture_regress, blobs=_fixture_blobs, spd=_fixture_spd,
                sample=_fixture_sample)


# ---------------------------------------------------------------------------------------------------- RECIPES
# Each recipe takes (ml, X, y, devices) and returns [(name, array), ...]: only public estimators, the public
# parallel entries and their documented attributes. A fresh estimator per arm.
def _svm(ml, X, y, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_svm(ml.SVC(kernel="rbf"), X, y, devices=devices)
    return [("dual_coef_", est.dual_coef_), ("intercept_", est.intercept_),
            ("decision", pc.predict_svm(est, X, devices=devices, method="decision_function"))]


def _logistic(ml, X, y, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_logistic(ml.LogisticRegression(), X, y, devices=devices)
    return [("coef_", est.coef_), ("intercept_", est.intercept_)]


def _lasso(ml, X, y, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_coordinate_descent(ml.Lasso(alpha=0.01), X, y, devices=devices)
    return [("coef_", est.coef_), ("intercept_", est.intercept_)]


def _ridge(ml, X, y, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_gram_estimator(ml.Ridge(alpha=1.0), X, y, devices=devices)
    return [("coef_", est.coef_), ("intercept_", est.intercept_)]


def _kmeans(ml, X, y, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_kmeans(ml.KMeans(n_clusters=4, random_state=0), X, devices=devices)
    return [("cluster_centers_", est.cluster_centers_), ("labels_", est.labels_)]


def _dbscan(ml, X, y, devices):
    from mojolearn import parallel_classical as pc
    # algorithm='brute': its eps-neighborhood queries go through dbscan/impl/multi_gpu.mojo; the default 'rbc'
    # bitmap passes on NVIDIA/AMD identical have no device plan (see THE LIST).
    est = pc.fit_dbscan(ml.DBSCAN(eps=0.5, min_samples=5, algorithm="brute"), X, devices=devices)
    return [("labels_", est.labels_)]


def _gmm(ml, X, y, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_gaussian_mixture(ml.GaussianMixture(n_components=3, random_state=0), X, devices=devices)
    return [("means_", est.means_),
            ("predict_proba", pc.predict_gaussian_mixture(est, X, devices=devices, method="predict_proba"))]


def _bootstrap(ml, X, y, devices):
    from mojolearn import parallel_classical as pc
    res = pc.bootstrap(X, devices=devices, n_resamples=999, random_state=0)
    return [("distribution", res.distribution), ("point_estimate", [res.point_estimate])]


def _cholesky(ml, X, y, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_cholesky(ml.Cholesky(), X, devices=devices)
    return [("L_", est.L_), ("solve", pc.solve_cholesky(est, y, devices=devices))]


def _iforest(ml, X, y, devices):
    from mojolearn import parallel_ensemble as pe
    est = pe.fit_isolation_forest(ml.IsolationForest(random_state=0), X, devices=devices)
    return [("score_samples", pe.score_isolation_forest(est, X, devices=devices, method="score_samples"))]


def _gbdt(ml, X, y, devices):
    from mojolearn import parallel_ensemble as pe
    est = pe.fit_boosting(ml.GradientBoostingRegressor(n_estimators=10), X, y, devices=devices)
    return [("predict", est.predict(X))]


def _forest(ml, X, y, devices):
    from mojolearn.parallel_ensemble import ParallelForestPredictor
    forest = ml.RandomForestClassifier(n_estimators=16, random_state=0,
                                       inference_engine="parallel_groves").fit(X, y)
    with ParallelForestPredictor(forest, devices=devices) as pool:
        return [("predict_proba", pool.predict_proba(X))]


RECIPES = dict(svm=_svm, logistic=_logistic, lasso=_lasso, ridge=_ridge, kmeans=_kmeans, dbscan=_dbscan,
               gmm=_gmm, bootstrap=_bootstrap, cholesky=_cholesky, iforest=_iforest, gbdt=_gbdt, forest=_forest)


# ---------------------------------------------------------------------------------------------------- WORKER
def worker(name, device, out_path):
    """One lane, every arm, in this (installed-wheel) interpreter; writes the lane's document to out_path."""
    doc = dict(lane=name, arms={}, error=None)
    try:
        row = {r[0]: r for r in LANES}[name]
        _, _family, recipe, fixture, rows, features, seed = row
        if os.environ.get(LOGICAL_ENV) != "1":
            raise RuntimeError("%s is not set: the logical arms would be refused" % LOGICAL_ENV)
        import numpy as np
        import mojolearn as ml
        from mojolearn._verify_distributed import digest
        doc.update(mojolearn_version=getattr(ml, "__version__", None), mojolearn_file=ml.__file__)
        X, y = FIXTURES[fixture](np, np.random.default_rng(seed), rows, features)
        for count in COUNTS:
            arm = dict(devices=list(devices_for(count, device)), parts=[], error=None)
            doc["arms"][str(count)] = arm
            started = time.time()
            ranks_take(sys.modules)  # clear what earlier arms (and imports) recorded
            try:
                for part, value in RECIPES[recipe](ml, X, y, devices_for(count, device)):
                    arm["parts"].append(dict(name=part, **digest(np.asarray(value))))
                arm["ranks"] = ranks_take(sys.modules)
            except Exception as error:  # fail closed: recorded, judged FAIL
                arm["error"] = "%s: %s" % (type(error).__name__, error)
                arm["parts"] = []
            arm["seconds"] = round(time.time() - started, 3)
    except Exception as error:
        doc["error"] = "%s: %s" % (type(error).__name__, error)
    Path(out_path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


def worker_env(device):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME") and not k.endswith("_DEVICE_COUNT")}
    env[LOGICAL_ENV] = "1"
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_lane(python, name, device, out, timeout):
    """Run one lane's worker in its own process (an external cwd, isolated mode: the installed wheel, never the
    source tree); returns the lane's (verdict, reason, digests, document)."""
    work = Path(out) / "lanes"
    work.mkdir(parents=True, exist_ok=True)
    result, log = work / (name + ".json"), work / (name + ".log")
    if result.exists():
        result.unlink()
    argv = [python, "-I", str(Path(__file__).resolve()), "--worker", name, "--device", str(device),
            "--result", str(result)]
    try:
        with log.open("w") as fh:
            proc = subprocess.run(argv, cwd=str(work), env=worker_env(device), stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        return "FAIL", "timed out after %ss" % timeout, {}, None
    try:
        doc = json.loads(result.read_text())
    except (OSError, ValueError):
        return "FAIL", "worker exited %s without a result (see lanes/%s.log)" % (rc, name), {}, None
    verdict, reason, digests = lane_verdict(name, doc)
    if verdict == "MATCH" and rc != 0:
        verdict, reason = "FAIL", "worker exited %s" % rc
    return verdict, reason, digests, doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--python", default=sys.executable, help="the interpreter the exact wheels are installed in")
    ap.add_argument("--vendor", default="", help="nvidia | amd (recorded)")
    ap.add_argument("--device", type=int, default=0, help="the one physical device every rank runs on")
    ap.add_argument("--out", default="")
    ap.add_argument("--jobs", type=int, default=0, help="lanes at once (default: every lane, capped at the cores)")
    ap.add_argument("--timeout", type=int, default=LANE_TIMEOUT, help="seconds per lane (every arm)")
    ap.add_argument("--worker", default="", help=argparse.SUPPRESS)
    ap.add_argument("--result", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        return worker(a.worker, a.device, a.result)
    if not a.out:
        ap.error("--out is required")
    errors = lane_errors()
    if errors:
        print("SHARDCHECK list refused: " + "; ".join(errors))
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = a.jobs or min(len(LANES), os.cpu_count() or 1)
    started = time.time()
    lanes = {}
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_lane, a.python, name, a.device, out, a.timeout): name for name in lane_names()}
        for future in cf.as_completed(futures):
            name = futures[future]
            try:
                verdict, reason, digests, doc = future.result()
            except Exception as error:
                verdict, reason, digests, doc = "FAIL", one_line("%s: %s" % (type(error).__name__, error)), {}, None
            arms = (doc or {}).get("arms") or {}
            lanes[name] = dict(verdict=verdict, reason=reason, digests={str(c): d for c, d in digests.items()},
                               ranks={str(c): (arms.get(str(c)) or {}).get("ranks") for c in COUNTS},
                               family={r[0]: r[1] for r in LANES}[name],
                               mojolearn_file=(doc or {}).get("mojolearn_file"))
    for name in lane_names():
        row = lanes[name]
        print(shardcheck_line(name, row["verdict"], row["reason"], {int(c): d for c, d in row["digests"].items()}))
    versions = sorted({str(r.get("mojolearn_file")) for r in lanes.values() if r.get("mojolearn_file")})
    doc = dict(schema=SCHEMA, vendor=a.vendor, device=a.device, counts=list(COUNTS), jobs=jobs,
               python=a.python, mojolearn_files=versions, seconds=round(time.time() - started, 1), lanes=lanes)
    doc["ok"] = not summary_errors(dict(doc, ok=True))
    (out / SUMMARY).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    bad = [n for n, r in lanes.items() if r["verdict"] != "MATCH"]
    print("SHARDCHECK summary %d/%d MATCH%s (%s)" % (len(lanes) - len(bad), len(lanes),
                                                     (", not: " + " ".join(sorted(bad))) if bad else "",
                                                     out / SUMMARY))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
