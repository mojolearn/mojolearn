#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact A/B gate for the default-off GBDT device winner fold.

Only Taxi and Istella-S are accepted.  One ``fit`` invocation is one fresh
process: it performs an untimed full-shape warmup followed by five retained
fits.  ``summarize`` requires three alternating-process records per A/B arm
and one full-shape sabotage record for each dataset/grow-policy cell.
"""
import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, os.path.join(ROOT, "python")):
    if path not in sys.path:
        sys.path.insert(0, path)

import speed_gbdt_arm as spec  # noqa: E402

DATASETS = ("taxi", "istella")
POLICIES = ("Depthwise", "Lossguide")
EXPECTED_SOURCE_SHA256 = {
    "taxi": "10d5d35f376a5b2ad5c66fabe624ee2caafb58bc5e7516824f801de9aab6cc15",
    "istella": "31f042376c0b819fe169cbbd998e840567c23dae25c14cb9054b460292f6ffef",
}
MIN_RETAINED = 5
REQUIRED_OUTERS = 3
SPREAD_LIMIT = 1.10
RATIO_LIMIT = 0.98


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_array(value):
    a = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def binding_record(model):
    path = os.path.abspath(model._bind("_mojolearn_gbdt").__file__)
    with open(path, "rb") as fh:
        sha = digest_bytes(fh.read())
    st = os.stat(path)
    return {"path": path, "bytes": st.st_size, "sha256": sha}


def estimator(policy):
    import mojolearn
    kw = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
              l2_leaf_reg=1.0, border_count=254, random_state=7,
              bootstrap_type="No", grow_policy=policy, loss="Logloss")
    if policy == "Lossguide":
        kw["max_leaves"] = 64
    return mojolearn.GradientBoosting(**kw)


def outputs(model, x_test, y_test):
    pred = np.asarray(model.predict(x_test))
    proba = np.asarray(model.predict_proba(x_test))
    curve = np.asarray(model.loss_curve_, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise RuntimeError("expected binary probabilities, got %r" % (proba.shape,))
    labels = np.argmax(proba, axis=1)
    target = np.asarray(y_test, dtype=np.int64)
    p = np.clip(proba[np.arange(target.size), target], 1e-300, 1.0)
    quality = {
        "accuracy": float(np.mean(labels == target)),
        "logloss": float(-np.mean(np.log(p.astype(np.float64)))),
        "rows": int(target.size),
    }
    text = str(model.model_).encode("utf-8")
    return {
        "model": digest_bytes(text),
        "model_bytes": len(text),
        "predict": digest_array(pred),
        "proba": digest_array(proba),
        "loss_curve": digest_array(curve),
    }, quality


def cmd_fit(args):
    if args.source_sha256 != EXPECTED_SOURCE_SHA256[args.dataset]:
        raise SystemExit("refusing unpinned %s source sha256 %s" %
                         (args.dataset, args.source_sha256))
    data = spec.load_with_fallback(args.dataset, "shipped", args.rows or None)
    x = np.ascontiguousarray(data.X_train, dtype=np.float32)
    y = np.ascontiguousarray(data.y_train, dtype=np.float32)
    xt = np.ascontiguousarray(data.X_test, dtype=np.float32)
    yt = np.ascontiguousarray(data.y_test, dtype=np.float32)
    input_record = {
        "source_path": os.path.abspath(args.source_path),
        "source_sha256": args.source_sha256,
        "x_shape": list(x.shape), "x_dtype": str(x.dtype), "x_sha256": digest_array(x),
        "y_shape": list(y.shape), "y_dtype": str(y.dtype), "y_sha256": digest_array(y),
        "x_test_shape": list(xt.shape), "x_test_dtype": str(xt.dtype),
        "x_test_sha256": digest_array(xt),
        "y_test_shape": list(yt.shape), "y_test_dtype": str(yt.dtype),
        "y_test_sha256": digest_array(yt),
    }
    if args.warmup:
        warm = estimator(args.policy)
        warm.fit(x, y)
        del warm
    times, hashes, qualities = [], [], []
    binding = None
    for retained in range(args.rounds):
        model = estimator(args.policy)
        t0 = time.perf_counter()
        model.fit(x, y)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if binding is None:
            binding = binding_record(model)
        out, quality = outputs(model, xt, yt)
        times.append(elapsed)
        hashes.append(out)
        qualities.append(quality)
        print("GDW FIT dataset=%s policy=%s arm=%s outer=%d retained=%d ms=%.3f model=%s"
              % (args.dataset, args.policy, args.arm, args.outer, retained,
                 elapsed, out["model"][:16]), flush=True)
    record = {
        "dataset": args.dataset, "policy": args.policy, "arm": args.arm,
        "outer": args.outer, "launch_position": args.launch_position,
        "warmup": bool(args.warmup),
        "numeric_mode": os.environ.get("MOJOLEARN_NUMERIC_MODE", ""),
        "commit": args.commit, "input": input_record, "binding": binding,
        "times_ms": times, "hashes": hashes, "quality": qualities,
    }
    with open(args.json, "w") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)


def _same(items):
    return len({json.dumps(x, sort_keys=True) for x in items}) == 1


def cmd_summarize(args):
    records = [json.load(open(p)) for pat in args.files
               for p in sorted(glob.glob(pat))]
    expected_cells = {(d, p) for d in DATASETS for p in POLICIES}
    cells = {}
    for rec in records:
        key = (rec["dataset"], rec["policy"])
        if key not in expected_cells:
            raise SystemExit("unexpected cell %r" % (key,))
        cells.setdefault(key, {}).setdefault(rec["arm"], []).append(rec)
    failures, summary = [], []
    if set(cells) != expected_cells:
        failures.append("cells=%r expected=%r" % (sorted(cells), sorted(expected_cells)))
    for key in sorted(expected_cells):
        arms = cells.get(key, {})
        row = {"dataset": key[0], "policy": key[1], "arms": {}}
        for arm in ("baseline", "candidate"):
            recs = arms.get(arm, [])
            outer = sorted(r.get("outer") for r in recs)
            expected_positions = ({1: 0, 2: 1, 3: 0} if arm == "baseline"
                                  else {1: 1, 2: 0, 3: 1})
            retained_ok = (outer == list(range(1, REQUIRED_OUTERS + 1)) and
                           all(len(r.get("times_ms", [])) >= MIN_RETAINED for r in recs) and
                           all(r.get("launch_position") == expected_positions.get(r["outer"])
                               for r in recs))
            times = [v for r in recs for v in r.get("times_ms", [])]
            outer_medians = [statistics.median(r["times_ms"]) for r in recs]
            spread = max(times) / min(times) if times and min(times) > 0 else float("inf")
            record_spreads = [max(r["times_ms"]) / min(r["times_ms"])
                              if r.get("times_ms") and min(r["times_ms"]) > 0
                              else float("inf") for r in recs]
            hashes = [h for r in recs for h in r.get("hashes", [])]
            qualities = [q for r in recs for q in r.get("quality", [])]
            stable = spread <= SPREAD_LIMIT and all(v <= SPREAD_LIMIT for v in record_spreads)
            valid = bool(recs) and retained_ok and stable and _same(hashes) and _same(qualities)
            row["arms"][arm] = {
                "records": len(recs), "retained": len(times), "spread": spread,
                "record_spreads": record_spreads, "outer_medians_ms": outer_medians,
                "stable": stable,
                "internally_exact": _same(hashes) and _same(qualities), "valid": valid,
            }
            if not valid:
                failures.append("%s/%s %s invalid" % (key[0], key[1], arm))
        b, c = arms.get("baseline", []), arms.get("candidate", [])
        provenance_ok = (bool(b and c) and
                         all(r.get("numeric_mode") == "identical" for r in b + c) and
                         _same([r.get("commit") for r in b + c]) and
                         _same([r["binding"]["sha256"] for r in b]) and
                         _same([r["binding"]["sha256"] for r in c]) and
                         b[0]["binding"]["sha256"] != c[0]["binding"]["sha256"])
        ab_exact = bool(b and c) and _same(
            [r["input"] for r in b + c]) and _same(
            [h for r in b + c for h in r["hashes"]]) and _same(
            [q for r in b + c for q in r["quality"]])
        row["ab_exact"] = ab_exact
        row["provenance_ok"] = provenance_ok
        bm = row["arms"]["baseline"]["outer_medians_ms"]
        cm = row["arms"]["candidate"]["outer_medians_ms"]
        ratio = max(cm) / min(bm) if bm and cm and min(bm) > 0 else float("inf")
        row["conservative_ratio"] = ratio
        sabotage = arms.get("sabotage", [])
        sabotage_reached = (len(sabotage) == 1 and sabotage[0].get("outer") == 1
                             and sabotage[0].get("launch_position") == 0
                             and sabotage[0].get("warmup") is False
                             and len(sabotage[0].get("times_ms", [])) == 1
                             and len(sabotage[0].get("hashes", [])) == 1
                             and b and all(sabotage[0]["hashes"][0][name] != b[0]["hashes"][0][name]
                                           for name in ("model", "predict", "proba", "loss_curve"))
                             and sabotage[0]["input"] == b[0]["input"]
                             and sabotage[0].get("numeric_mode") == "identical"
                             and sabotage[0]["binding"]["sha256"] not in
                             (b[0]["binding"]["sha256"], c[0]["binding"]["sha256"]))
        row["sabotage_reached"] = sabotage_reached
        row["pass"] = (row["arms"]["baseline"]["valid"] and
                       row["arms"]["candidate"]["valid"] and ab_exact and provenance_ok and
                       sabotage_reached and ratio <= RATIO_LIMIT)
        if not provenance_ok:
            failures.append("%s/%s binding or commit provenance invalid" % key)
        if not ab_exact:
            failures.append("%s/%s A/B bytes or quality differ" % key)
        if not sabotage_reached:
            failures.append("%s/%s sabotage did not move all outputs" % key)
        if ratio > RATIO_LIMIT:
            failures.append("%s/%s conservative ratio %.6f > %.2f" %
                            (key[0], key[1], ratio, RATIO_LIMIT))
        summary.append(row)
        print("GDW GATE dataset=%s policy=%s ratio=%.6f exact=%s provenance=%s sabotage=%s pass=%s"
              % (key[0], key[1], ratio, ab_exact, provenance_ok,
                 sabotage_reached, row["pass"]))
    result = {"pass": not failures and all(r["pass"] for r in summary),
              "limits": {"min_retained": MIN_RETAINED, "outers": REQUIRED_OUTERS,
                         "spread": SPREAD_LIMIT, "ratio": RATIO_LIMIT},
              "cells": summary, "failures": failures}
    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    if failures:
        print("GDW REJECT " + "; ".join(failures))
        return 1
    print("GDW PROMOTE all four Taxi/Istella Depthwise/Lossguide gates passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--dataset", choices=DATASETS, required=True)
    fit.add_argument("--policy", choices=POLICIES, required=True)
    fit.add_argument("--arm", choices=("baseline", "candidate", "sabotage"), required=True)
    fit.add_argument("--outer", type=int, required=True)
    fit.add_argument("--launch-position", type=int, choices=(0, 1), required=True)
    fit.add_argument("--rounds", type=int, default=MIN_RETAINED)
    fit.add_argument("--rows", type=int, default=0)
    fit.add_argument("--warmup", type=int, choices=(0, 1), default=1)
    fit.add_argument("--source-path", required=True)
    fit.add_argument("--source-sha256", required=True)
    fit.add_argument("--commit", required=True)
    fit.add_argument("--json", required=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("files", nargs="+")
    summ.add_argument("--json", required=True)
    args = ap.parse_args()
    if args.cmd == "fit":
        cmd_fit(args)
        return 0
    return cmd_summarize(args)


if __name__ == "__main__":
    sys.exit(main())
