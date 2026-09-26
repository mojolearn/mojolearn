# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The benchmark board's INFERENCE cells (tools/bench_board.py loads this).

The fit cells time training. These time batch prediction, each arm with ITS
OWN model from the same race's fit rounds (no fit is retimed), on the same
rows, in the same output kind, raced round by round like the fit:

  trees      bench/speed/forest_speed_arm.py --infer. Batches `test` (the
             held-out split FSPEED-ACC scores) and `large` (the first
             1,000,000 training rows, capped at the training rows). Lines
             FSPEED-INFER*, parsed here. Each arm's call is on its
             FSPEED-INFER-PATH line.
  classical  tools/classical_two_datasets.py race --infer, lanes kmeans
             (predict), pca (transform), ols (predict), svc (predict) on the
             eval rows Xq; the race JSON's `infer` record. knn and kde already
             time inference (kneighbors, score_samples) as their race; dbscan
             and hdbscan have no predict.

The cells go to a race record's `infer_cells`, never to `cells`, so every fit
table, the coverage count of fit cells and "Quality at a glance" read exactly
what they read before. Ratios are computed per batch, ours over each
opponent, never FAST over IDENTICAL. Nothing here states a direction.

This module imports nothing from bench_board at import time; the board passes
its own helpers in (`bb`, a namespace of bench_board's globals).
"""
import json
import re
import statistics

TREE_BATCHES = ("test", "large")
CLASSICAL_INFER_LANES = ("kmeans", "pca", "ols", "svc")
CLASSICAL_CALL = {"kmeans": "predict", "pca": "transform", "ols": "predict", "svc": "predict"}

#: What each trees arm's inference clock covers, for the board's settings and
#: docs (the driver's FSPEED-INFER-PATH line is the run's own record).
TREE_PATHS = {
    "ours": "predict_proba(X) column 1 (gbdt, rf, et); score_samples(X) (iforest, which "
            "rebuilds its forest inside every scoring call, DEVIATION 874)",
    "catboost": "predict_proba(X, task_type GPU on the -gpu arm, CPU otherwise)",
    "xgboost": "Booster.inplace_predict (no DMatrix); on a CUDA booster the rows go up as a "
               "cupy array and the result comes back inside the clock",
    "lightgbm": "Booster.predict(X) (LightGBM predicts on the CPU)",
    "scikit-learn": "predict_proba(X) or score_samples(X), n_jobs -1",
    "cuml": "RandomForest converted once to FIL outside the clock, then FIL predict_proba on "
            "host rows; IsolationForest.score_samples",
}

NOT_COVERED = (
    "Inference, trees: categorical (criteo) frames are not wired into the inference phase; "
    "a single-row latency batch is not timed (the batches are the held-out split and 1,000,000 "
    "training rows); ONNX, Treelite and other export paths are not raced.",
    "Inference, classical: the classical2 family's predict calls (the linear models, "
    "GaussianMixture, SVR, KernelRidge and others in tools/bench_board_more.py) are timed as "
    "those lanes define their clocks, not as a separate inference cell; svc times predict, "
    "not decision_function.",
)


def _kv(text):
    rx = re.compile(r"(\S+?)=(.*?)(?=\s+\S+?=|$)")
    return {m.group(1): m.group(2).strip() for m in rx.finditer(text.strip())}


def driver_args(race):
    """Extra driver arguments for this race when inference is on."""
    if race["family"] == "trees":
        return ["--infer"]
    if race["family"] == "classical" and race["lane"] in CLASSICAL_INFER_LANES:
        return ["--infer"]
    return []


def plan_cells(race):
    """(arm, batch) pairs of the inference cells this race will produce."""
    if race["family"] == "trees":
        return [(a, b) for b in TREE_BATCHES for a in race["arms"]]
    if race["family"] == "classical" and race["lane"] in CLASSICAL_INFER_LANES:
        return [(a, "Xq") for a in race["arms"]]
    return []


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_tree_infer(path):
    """The FSPEED-INFER* lines of one forest_speed_arm.py log."""
    out = {"paths": {}, "rounds": {}, "warmup": {}, "acc": {}, "refused": {}, "agree": {},
           "notes": []}
    try:
        fh = open(path, errors="replace")
    except OSError:
        return out
    with fh:
        for line in fh:
            head, _, rest = line.rstrip("\n").partition(" ")
            if not head.startswith("FSPEED-INFER"):
                continue
            f = _kv(rest)
            arm, batch = f.get("arm"), f.get("batch")
            if head == "FSPEED-INFER-PATH" and arm:
                out["paths"][arm] = f.get("call", "")
            elif head == "FSPEED-INFER" and arm and batch:
                rec = out["rounds"].setdefault((arm, batch), {"ms": [], "hashes": [], "rows": None})
                try:
                    rec["ms"].append(float(f["ms"]))
                    rec["rows"] = int(f["rows"])
                except (KeyError, ValueError):
                    pass
                if f.get("hash") not in (None, "-", "None"):
                    rec["hashes"].append(f["hash"])
            elif head == "FSPEED-INFER-WARMUP" and arm and batch:
                try:
                    out["warmup"][(arm, batch)] = (float(f["ms"]), int(f["rows"]))
                except (KeyError, ValueError):
                    pass
            elif head == "FSPEED-INFER-ACC" and arm and batch:
                try:
                    out["acc"].setdefault((arm, batch), {})[f["metric"]] = float(f["value"])
                except (KeyError, ValueError):
                    pass
            elif head == "FSPEED-INFER-REFUSED" and arm:
                out["refused"][(arm, batch or "all")] = f.get("reason", "")[:200]
            elif head == "FSPEED-INFER-AGREE" and batch:
                out["agree"][batch] = f
            elif head == "FSPEED-INFER-NOTE":
                out["notes"].append(rest[:300])
    return out


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def _timing(cell, ms, rounds, refused, bb):
    cell.update(times_ms=ms, median_ms=float(statistics.median(ms)) if ms else None,
                min_ms=min(ms) if ms else None, max_ms=max(ms) if ms else None,
                rounds=len(ms), status=bb._status(ms, refused, rounds))


def _ratios(cells, bb):
    groups = {}
    for c in cells:
        groups.setdefault(c["batch"], []).append(c)
    for g in groups.values():
        bb.add_ratios(g)
    return cells


def tree_cells(bb, ctx, race, log_path, fit_cells):
    """Inference cells of one trees race, from its log."""
    p = parse_tree_infer(log_path)
    fit_q = {c["arm"]: (c.get("quality") or {}) for c in fit_cells}
    fit_verdict = next((c.get("verdict") for c in fit_cells if c.get("verdict")), "UNKNOWN")
    cells = []
    names = list(race["arms"]) + sorted({a for a, _ in p["rounds"]} - set(race["arms"]))
    for batch in TREE_BATCHES:
        agree = p["agree"].get(batch) or {}
        for arm in names:
            mode = race["our_arms"].get(arm)
            cell = bb.base_cell(ctx, race, arm, mode)
            cell.update(phase="infer", batch=batch, call=p["paths"].get(arm))
            rec = p["rounds"].get((arm, batch)) or {"ms": [], "hashes": [], "rows": None}
            warm = p["warmup"].get((arm, batch))
            refused = p["refused"].get((arm, batch)) or p["refused"].get((arm, "all"))
            _timing(cell, rec["ms"], ctx["rounds"], refused, bb)
            if not rec["ms"] and not refused:
                cell["status"] = "UNKNOWN(no inference lines)"
            cell["batch_rows"] = rec["rows"] or (warm[1] if warm else None)
            cell["warmup_ms"] = warm[0] if warm else None
            h = rec["hashes"]
            cell["hash"] = h[-1] if h else None
            cell["hash_stable"] = (len(set(h)) == 1) if h else None
            q = dict(p["acc"].get((arm, batch)) or {})
            for m, v in list(q.items()):
                fv = fit_q.get(arm, {}).get(m)
                if isinstance(fv, (int, float)):
                    q["%s_matches_fit" % m] = abs(fv - v) <= 1e-6 * max(1.0, abs(fv))
            if arm == "ours-ab" and agree:
                q["bits_equal_vs_ours_identical"] = agree.get("bits_equal") == "yes"
                try:
                    q["max_abs_diff_vs_ours_identical"] = float(agree.get("max_abs_diff"))
                except (TypeError, ValueError):
                    pass
            cell["quality"] = q
            cell["verdict"] = fit_verdict
            cell["comparability"] = {"fit_verdict": fit_verdict,
                                     "clock": "host rows in, host predictions out"}
            cells.append(cell)
    return _ratios(cells, bb)


def classical_cells(bb, ctx, race, r):
    """Inference cells of one classical race, from its JSON's `infer` record."""
    inf = (r or {}).get("infer")
    if not inf:
        return []
    fit = {c: a for c, a in ((r.get("arms") or {}).items())}
    qual = inf.get("quality") or {}
    cells = []
    for arm in list(race["arms"]) + [a for a in inf.get("arms", {}) if a not in race["arms"]]:
        a = (inf.get("arms") or {}).get(arm)
        mode = race["our_arms"].get(arm)
        cell = bb.base_cell(ctx, race, arm, mode)
        cell.update(phase="infer", batch=inf.get("batch", "Xq"), batch_rows=inf.get("rows"))
        if a is None:
            cell["status"] = "UNKNOWN(not in race json)"
            cells.append(cell)
            continue
        refused = None if a.get("status") == "ok" else "%s: %s" % (
            a.get("status"), json.dumps(a.get("error"))[:200])
        _timing(cell, a.get("ms") or [], ctx["rounds"], refused, bb)
        cell["warmup_ms"] = a.get("warmup_ms")
        cell["call"] = a.get("call_text") or "%s(Xq)" % inf.get("call")
        cell["hash"] = (a.get("digests") or [None])[-1]
        cell["hash_stable"] = a.get("digest_stable")
        cell["quality"] = {k: v for k, v in (qual.get(arm) or {}).items() if k != "error"}
        span = ((fit.get(arm) or {}).get("span") or {})
        info = a.get("info") or {}
        ours_span = ((fit.get("ours") or {}).get("span") or {})
        asym = []
        if bb.arm_library(arm) != "mojolearn" and span.get("input_home") == "device" \
                and ours_span.get("input_home") == "host":
            asym.append("upload_outside_its_clock")
        cell["verdict"] = ("SPAN-ASYMMETRIC(%s)" % "+".join(asym)) if asym else "LIKE-FOR-LIKE-SPAN"
        cell["comparability"] = {"span": span, "span_asymmetry": asym,
                                 "upload_ms_untimed": info.get("upload_ms_untimed")}
        cells.append(cell)
    return _ratios(cells, bb)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_race(bb, rr):
    """The inference table under one race's fit table ([] when none)."""
    ic = rr.get("infer_cells") or []
    if not ic:
        return []
    L = ["", "Inference (each arm predicts with its own model from the fit rounds above):", "",
         "| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | "
         "ours FAST / arm | quality | hash stable | comparability | status |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in ic:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            bb._arm_label(c), c.get("batch"), bb._f(c.get("batch_rows")), bb._f(c["median_ms"]),
            "%s..%s" % (bb._f(c["min_ms"]), bb._f(c["max_ms"])) if c["min_ms"] is not None else "-",
            c["rounds"], bb._f(c.get("ratio_ours_identical_over"), 3),
            bb._f(c.get("ratio_ours_fast_over"), 3), bb._q(c.get("quality")),
            bb._f(c.get("hash_stable")), bb.clean(c.get("verdict")), bb.clean(c["status"])))
    calls = []
    for c in ic:
        if c.get("call") and (c["arm"], c["call"]) not in calls:
            calls.append((c["arm"], c["call"]))
    for arm, call in calls:
        L.append("")
        L.append("inference call, %s: %s" % (bb.clean(arm), bb.clean(call)))
    return L


def render_glance(bb, races):
    """'Inference at a glance': per race and batch, our medians, whether our
    FAST and IDENTICAL predictions agree bit for bit, each opponent's median
    and ratio."""
    rows = []
    for rid in sorted(races):
        ic = races[rid].get("infer_cells") or []
        for batch in sorted({c["batch"] for c in ic}, key=lambda b: (b != "test", b)):
            g = [c for c in ic if c["batch"] == batch]
            fast = next((c for c in g if c["library"] == "mojolearn" and c["mode"] == "fast"), None)
            ident = next((c for c in g if c["library"] == "mojolearn" and c["mode"] == "identical"), None)
            agree = "-"
            if fast is not None:
                q = fast.get("quality") or {}
                v = q.get("bits_equal_vs_ours_identical", q.get("bits_equal_vs_ours"))
                agree = bb._f(v) if v is not None else "-"
            opps = [c for c in g if c["library"] != "mojolearn"]
            rows.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                races[rid]["family"], races[rid]["lane"], races[rid]["dataset"], batch,
                bb._f((ident or fast or {}).get("batch_rows")),
                bb._f(fast["median_ms"]) if fast else "-",
                bb._f(ident["median_ms"]) if ident else "-", agree,
                "; ".join("%s %s ms (IDENTICAL/arm %s)" % (
                    c["arm"], bb._f(c["median_ms"]), bb._f(c.get("ratio_ours_identical_over"), 3))
                    for c in opps) or "-"))
    if not rows:
        return []
    return ["## Inference at a glance", "",
            "Batch prediction, each arm with its own fitted model from the same race; medians in "
            "ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows.", "",
            "| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | "
            "FAST = IDENTICAL bits | opponents |",
            "|---|---|---|---|---|---|---|---|---|"] + rows + [""]


def coverage(races):
    st = {}
    n = 0
    for r in races.values():
        for c in r.get("infer_cells") or []:
            n += 1
            k = c["status"].split("(")[0]
            st[k] = st.get(k, 0) + 1
    if not n:
        return None
    return "Inference cells: %d (%s)." % (n, ", ".join("%s %d" % kv for kv in sorted(st.items())))
