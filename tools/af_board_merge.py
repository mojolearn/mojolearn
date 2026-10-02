#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The M3 FAST board refresh: lane selection, box-side helpers, and the merge.

    af_board_merge.py merge --logs LOGFILE [--tsv TSV] [--trees-board JSON] [--out MD]
        Laptop. LOGFILE holds the `lq log m3 <tag> '^AFB'` output of every job
        tools/af_board_queue.sh queued (AFB result lines and the opp job's
        AFB-OPPZ chunks). Writes docs/apple-fast/BOARD_M3_FAST.md: per lane x
        dataset, FAST before (0.8.34 M3 board; Sept 29 M3 board for trees) ->
        after, best opponent and arm, ratio before -> after, held-out quality,
        flips. Worst ratio after first. Opponents are not re-raced.

    af_board_merge.py select --branch BR [--lanesel JSON | --lanes a,b] [--batch N] [--prefix P] [--lq LQ]
        Laptop (called by af_board_queue.sh). Prints the lq lines.
    af_board_merge.py builds FAMILY lane:ds,...
        Box. The FAST build scripts (bindings/<name>.sh without .sh) a batch needs.
    af_board_merge.py extract BOARD_JSON
        Box. AFB-OPPZ lines: the board's FAST and best-opponent cells, gzip+base64.
"""
import argparse
import base64
import gzip
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EVID = os.path.expanduser("~/mojolearn-evidence")
TSV = os.path.join(EVID, "board-0834-times.tsv")
TREES_BOARD = os.path.join(EVID, "board-archive/m3-ultra/2026-09-29_m3ultra_checked/board.json")
OUT_MD = os.path.join(REPO, "docs/apple-fast/BOARD_M3_FAST.md")

# Tree board lanes (bench/speed/forest_speed_arm.py) -> binding build, datasets.
TREE_LANES = {
    "gbdt-symmetric": ("build_gbdt", ("taxi", "istella")),
    "gbdt-symmetric-1000": ("build_gbdt", ("taxi", "istella")),
    "gbdt-depthwise": ("build_gbdt", ("taxi", "istella")),
    "gbdt-lossguide": ("build_gbdt", ("taxi", "istella")),
    "gbdt-multiclass": ("build_gbdt", ("taxi", "istella")),
    "gbdt-ordered": ("build_gbdt", ("taxi", "istella")),
    "gbdt-categorical": ("build_gbdt", ("taxi",)),
    "gbdt-rank-yetirank": ("build_gbdt", ("istella",)),
    "gbdt-rank-pairlogit": ("build_gbdt", ("istella",)),
    "rf": ("build_rf", ("taxi", "istella")),
    "et": ("build_trees", ("taxi", "istella")),
    "iforest": ("build_svm", ("taxi", "istella")),
}
# classical_two_datasets.py lanes -> the binding their estimator answers from.
CTD_BINDING = {"kmeans": "_mojolearn", "pca": "_mojolearn_estimators", "ols": "_mojolearn_estimators",
               "knn": "_mojolearn", "kde": "_mojolearn_estimators", "svc": "_mojolearn_svm",
               "dbscan": "_mojolearn_estimators", "hdbscan": "_mojolearn_hdbscan"}
FAMILIES = ("trees", "algos", "classical2", "classical")
BOARD_DATASETS = ("taxi", "istella")
# Identity-registry lane name (normalized) -> board lane, where the names differ.
ALIASES = {"bayes-ridge": "bayesian-ridge", "gbdt-yeti-rank": "gbdt-rank-yetirank",
           "gbdt-pair-logit": "gbdt-rank-pairlogit", "dt-clf": "decision-tree-clf",
           "dt-reg": "decision-tree-reg", "random-embedding": "random-trees-embedding",
           "shap-tree": "tree-shap", "shap-kernel": "kernel-shap", "shap-permutation": "permutation-shap",
           "logistic-cv": "logreg-cv", "logistic": "logreg", "onehot-encoder": "onehot",
           "ordinal-encoder": "ordinal", "polynomial-features": "poly-features",
           "spline-transformer": "spline", "kpca": "kernel-pca", "poly-sketch": "poly-count-sketch",
           "radius": "radius-neighbors", "gp": "gpr", "linear-svc": "linearsvc", "linear-svr": "linearsvr",
           "kmeans": "kmeans", "rf-clf": "rf", "rf-reg": "rf", "et-clf": "et", "et-reg": "et",
           "gbdt-categorical-ctr": "gbdt-categorical", "ridge-clf": "ridge-clf", "gram-pca": "pca",
           "gram-ols": "ols", "gram-tsvd": "tsvd", "decomp-ipca": "incremental-pca",
           "decomp-factor-analysis": "factor-analysis", "enet-cv": "enet-cv"}
AREAS = ("linear-", "neighbors-", "prep-", "decomp-", "cluster-", "metrics-")
#: A changed file that reaches more than --wide registry lanes is a shared
#: helper every binding imports (fast_mma_knn reaches the base binding, so
#: lane_select names ~350 lanes). Its lanes count only when their name holds a
#: token of the estimators that call it; a wide file with no entry here keeps
#: every lane it reaches (and the select output says so).
WIDE_TOKENS = {
    "neighbors/": ("knn", "neighbors", "radius", "nearest", "lof", "isomap", "lle", "tsne", "umap",
                   "spectral-embedding", "label-propagation", "label-spreading", "dbscan", "optics",
                   "hdbscan", "agglomerative", "mds", "connected-components"),
}


def _board_lanes():
    """family -> {lane: (binding build, datasets)} from the drivers' own tables."""
    sys.path.insert(0, HERE)
    import bench_board_algos as algos
    import bench_board_more as more
    out = {"trees": dict(TREE_LANES), "algos": {}, "classical2": {}, "classical": {}}
    rb = algos._READBACK_BINDING
    for lane, s in algos.LANES.items():
        b = s.get("binding") or rb.get(s["xlane"], "_mojolearn_x_" + s["xlane"])
        ds = tuple(d for d in s.get("datasets", BOARD_DATASETS) if d in BOARD_DATASETS)
        out["algos"][lane] = (_build_of(b), ds)
    for lane, (_, b, ds) in more.LANES.items():
        out["classical2"][lane] = (_build_of(b), tuple(d for d in ds if d in BOARD_DATASETS))
    for lane, b in CTD_BINDING.items():
        out["classical"][lane] = (_build_of(b), BOARD_DATASETS)
    return out


def _build_of(binding):
    name = binding[len("_mojolearn"):].lstrip("_")
    cands = ["build_" + name] if name else ["build"]
    if name.startswith("x_"):
        cands.append("build_" + name[2:])
    for c in cands:
        if os.path.exists(os.path.join(REPO, "bindings", c + ".sh")):
            return c
    return cands[0]


def _norm(r):
    for p in ("par-", "x-", "trees-"):
        if r.startswith(p):
            r = r[len(p):]
    for a in AREAS:
        if r.startswith(a):
            r = r[len(a):]
    return ALIASES.get(r, r)


def _matches(board_lane, reg):
    return reg == board_lane or reg.startswith(board_lane + "-") or board_lane.startswith(reg + "-")


def cmd_select(a):
    lanes = _board_lanes()
    forced = [x for x in (a.lanes or "").split(",") if x]
    regs = []
    if not forced:
        sel = json.load(open(a.lanesel))
        raw = set()
        for path, reached in (sel.get("by_path") or {}).items():
            if len(reached) <= a.wide:
                raw.update(reached)
                continue
            toks = next((t for pre, t in WIDE_TOKENS.items() if path.startswith(pre)), None)
            if toks is None:
                print("# wide file %s (%d lanes): no token entry, every lane kept" % (path, len(reached)))
                raw.update(reached)
                continue
            kept = [r for r in reached if any(t in r for t in toks)]
            print("# wide file %s: %d of %d lanes kept (callers %s)" % (path, len(kept), len(reached),
                                                                       ",".join(toks)))
            raw.update(kept)
        if not sel.get("by_path"):
            raw = set(sel.get("lanes", []))
        regs = sorted({_norm(r) for r in raw})
        print("# lane_select: %d registry lanes over %d changed files (%d inert); fallback=%s" % (
            len(sel.get("lanes", [])), len(sel.get("changed", [])), len(sel.get("inert", [])),
            sel.get("fallback")))
    batches = []
    for fam in FAMILIES:
        pairs = []
        for lane in sorted(lanes[fam]):
            _, ds = lanes[fam][lane]
            if forced:
                if lane not in forced and "%s/%s" % (fam, lane) not in forced:
                    continue
            elif not any(_matches(lane, r) for r in regs):
                continue
            pairs += ["%s:%s" % (lane, d) for d in ds]
        for i in range(0, len(pairs), a.batch):
            batches.append((fam, pairs[i:i + a.batch]))
    n = sum(len(p) for _, p in batches)
    print("# %d lane x dataset pairs in %d batches: %s" % (n, len(batches), ", ".join(
        "%s=%d" % (f, sum(len(p) for g, p in batches if g == f)) for f in FAMILIES)))
    tags = []
    for k, (fam, pairs) in enumerate(batches, 1):
        tag = "%s-%s-%d" % (a.prefix, fam, k)
        tags.append(tag)
        print("%s add m3 CMD %s %s 'bash tools/af_board_queue.sh run %s %s %s'" % (
            a.lq, a.branch, tag, tag, fam, ",".join(pairs)))
    tags.append(a.prefix + "-opp")
    print("%s add m3 CMD %s %s 'bash tools/af_board_queue.sh opp %s'" % (a.lq, a.branch, a.prefix + "-opp",
                                                                     a.prefix + "-opp"))
    # lq log keeps the last 40 matching lines: AFB lines per batch (<= batch + a few build lines),
    # AFB-OPPZ chunks one decade at a time.
    print("# merge once done (bash): f=~/mojolearn-evidence/apple-fast-board/afb.log; : > $f; "
          "for t in %s; do %s log m3 $t '^AFB(-BUILD| )' >> $f; done; "
          "for d in '' 1 2 3 4 5 6 7 8 9; do %s log m3 %s \"^AFB-OPPZ ${d}[0-9] \" >> $f; done; "
          "python3 tools/af_board_merge.py merge --logs $f"
          % (" ".join(tags[:-1]), a.lq, a.lq, tags[-1]))


def cmd_builds(a):
    lanes = _board_lanes()[a.family]
    need = []
    for p in a.pairs.split(","):
        b = lanes.get(p.split(":")[0], (None,))[0]
        if b and b not in need:
            need.append(b)
    print(" ".join(need))


def _q(q):
    if not isinstance(q, dict):
        return {}
    return {k: float("%.5g" % v) for k, v in q.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.endswith("_matches_fit")}


def _cells(rec):
    """(fast cell, best opponent cell) of one board race, fit phase."""
    fast, opp = None, []
    for c in rec.get("cells") or []:
        if c.get("phase") not in (None, "fit"):
            continue
        ms = c.get("median_ms")
        if c.get("library") == "mojolearn" or str(c.get("arm", "")).startswith("ours"):
            if c.get("mode") == "fast" and ms:
                fast = c
        elif ms and c.get("status") == "ok":
            opp.append(c)
    best = min(opp, key=lambda c: c["median_ms"]) if opp else None
    return fast, best


def cmd_extract(a):
    d = json.load(open(a.board))
    rows = {}
    for rid, rec in (d.get("races") or {}).items():
        if rec.get("family") in ("trees", "neural"):
            continue
        fast, best = _cells(rec)
        key = "%s/%s" % (rec.get("lane") or rid.split("/")[1], rec.get("dataset") or rid.split("/")[2])
        rows[key] = {"fast_ms": fast and round(fast["median_ms"], 1), "fast_q": _q(fast and fast.get("quality")),
                     "best_arm": best and best.get("arm"), "best_ms": best and round(best["median_ms"], 1),
                     "best_q": _q(best and best.get("quality"))}
    blob = base64.b64encode(gzip.compress(json.dumps(rows, separators=(",", ":")).encode())).decode()
    chunks = [blob[i:i + 340] for i in range(0, len(blob), 340)]
    if len(chunks) > 100:
        sys.exit("extract: %d chunks, more than the merge's 100-chunk read" % len(chunks))
    for i, c in enumerate(chunks):
        print("AFB-OPPZ %d %d %s" % (i, len(chunks), c))


def _read_logs(path):
    res, chunks, total = {}, {}, None
    for ln in open(path, errors="replace"):
        ln = ln.strip()
        m = re.match(r"^AFB-OPPZ (\d+) (\d+) (\S+)$", ln)
        if m:
            chunks[int(m.group(1))] = m.group(3)
            total = int(m.group(2))
            continue
        if not ln.startswith("AFB "):
            continue
        kv = dict(re.findall(r"(\w+)=(\[[^\]]*\]|\S+)", ln))
        key = (kv.get("family"), kv.get("lane"), kv.get("ds"))
        res[key] = kv  # later lines (a requeue) win
    opp = {}
    if total and len(chunks) == total:
        opp = json.loads(gzip.decompress(base64.b64decode("".join(chunks[i] for i in range(total)))))
    elif chunks:
        print("warning: AFB-OPPZ incomplete (%d of %s chunks); no 0.8.34 quality" % (len(chunks), total),
              file=sys.stderr)
    return res, opp


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _fmt_ms(v):
    return "-" if v is None else ("%.0f" % v if v >= 100 else "%.1f" % v)


def _fmt_r(v):
    return "-" if v is None else "%.2f" % v


def _fmt_q(q):
    if not q:
        return "-"
    if isinstance(q, str):
        return q.replace(",", ", ")
    return ", ".join("%s=%.4g" % (k, v) for k, v in sorted(q.items()) if not k.endswith("_matches_fit"))[:90]


def cmd_merge(a):
    res, opp = _read_logs(a.logs)
    tsv = {}
    if os.path.exists(a.tsv):
        lines = open(a.tsv).read().splitlines()
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            r = dict(zip(hdr, ln.split("\t")))
            tsv[(r["algo"], r["dataset"])] = r
    trees = {}
    if os.path.exists(a.trees_board):
        for rid, rec in json.load(open(a.trees_board)).get("races", {}).items():
            if rec.get("family") == "trees":
                trees[(rec["lane"], rec["dataset"])] = _cells(rec)
    rows = []
    for (fam, lane, ds), kv in res.items():
        after = _num(kv.get("median_ms"))
        q_after = kv.get("q") if kv.get("q") not in (None, "-") else None
        before = best = None
        best_arm, q_before, q_best, src = "-", None, None, "-"
        if fam == "trees":
            fast, b = trees.get((lane, ds), (None, None))
            if fast:
                before, q_before = fast["median_ms"], _q(fast.get("quality"))
            if b:
                best, best_arm, q_best = b["median_ms"], b.get("arm"), _q(b.get("quality"))
            src = "M3 2026-09-29" if (fast or b) else "-"
        else:
            t = tsv.get((lane, ds), {})
            before, best = _num(t.get("M3_fast_ms")), _num(t.get("M3_best_ms"))
            best_arm = t.get("M3_best_arm") or "-"
            o = opp.get("%s/%s" % (lane, ds)) or {}
            if before is None:
                before = _num(o.get("fast_ms"))
            if best is None and o.get("best_ms"):
                best, best_arm = _num(o.get("best_ms")), o.get("best_arm") or "-"
            q_before, q_best = o.get("fast_q"), o.get("best_q")
            src = "M3 0.8.34" if (t or o) else "-"
        r_before = before / best if (before and best) else None
        r_after = after / best if (after and best) else None
        flip = ""
        if r_before is not None and r_after is not None:
            if r_before > 1 >= r_after:
                flip = "FLIP faster"
            elif r_before <= 1 < r_after:
                flip = "FLIP slower"
        rows.append(dict(fam=fam, lane=lane, ds=ds, st=kv.get("status"), before=before, after=after,
                         best=best, best_arm=best_arm, rb=r_before, ra=r_after, qa=q_after, qb=q_before,
                         qo=q_best, flip=flip, src=src, head=kv.get("head"), digest=kv.get("digest"),
                         runs=kv.get("runs"), tag=kv.get("tag")))
    rows.sort(key=lambda r: (r["ra"] is None, -(r["ra"] or 0), r["lane"], r["ds"]))
    heads = sorted({r["head"] for r in rows if r["head"]})
    ok = [r for r in rows if r["ra"] is not None]
    flips_f = [r for r in rows if r["flip"] == "FLIP faster"]
    flips_s = [r for r in rows if r["flip"] == "FLIP slower"]
    gm = (statistics.geometric_mean([r["ra"] for r in ok]) if ok else None)
    out = ["# M3 FAST board refresh (lane/apple-fast)", "",
           "Our FAST arm on the M3 Ultra Metal GPU at head %s, 1 warm-up + 3 timed rounds at board size "
           "(rows-full; trees MOJOLEARN_SPEED_SIZE=shipped). Opponents are not re-raced: classical times come from "
           "the M3 0.8.34 board (`~/mojolearn-evidence/board-0834-times.tsv`, quality from its board.json), trees "
           "from the 2026-09-29 M3 board (older tree params on some lanes). Ratio = our FAST ms / best opponent "
           "ms; below 1 is faster. Rows sort worst ratio after first. Written by `tools/af_board_merge.py`."
           % (", ".join(heads) or "?"), "",
           "Summary: %d rows, %d with a ratio, %d faster than the best opponent after, geometric-mean ratio %s. "
           "Flips to faster: %s. Flips to slower: %s." % (
               len(rows), len(ok), sum(1 for r in ok if r["ra"] <= 1), _fmt_r(gm),
               ", ".join("%s %s" % (r["lane"], r["ds"]) for r in flips_f) or "none",
               ", ".join("%s %s" % (r["lane"], r["ds"]) for r in flips_s) or "none"), "",
           "| lane | dataset | family | FAST before ms | FAST after ms | best opponent | opp ms | ratio before | "
           "ratio after | flip | quality after (FAST) | quality before (FAST) | opponent quality | status |",
           "|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|"]
    for r in rows:
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["lane"], r["ds"], r["fam"], _fmt_ms(r["before"]), _fmt_ms(r["after"]), r["best_arm"],
            _fmt_ms(r["best"]), _fmt_r(r["rb"]), _fmt_r(r["ra"]), r["flip"] or "", _fmt_q(r["qa"]),
            _fmt_q(r["qb"]), _fmt_q(r["qo"]), r["st"]))
    out += ["", "Sources: before = %s; job tags %s." % (
        "M3 0.8.34 board (classical), M3 2026-09-29 board ours-ab FAST cells (trees)",
        ", ".join(sorted({r["tag"] for r in rows if r["tag"]})))]
    with open(a.out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote %s: %d rows, %d flips faster, %d flips slower, geomean ratio %s" % (
        a.out, len(rows), len(flips_f), len(flips_s), _fmt_r(gm)))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select")
    s.add_argument("--branch", required=True)
    s.add_argument("--lanesel")
    s.add_argument("--lanes")
    s.add_argument("--batch", type=int, default=18)
    s.add_argument("--wide", type=int, default=120)
    s.add_argument("--prefix", default="afb")
    s.add_argument("--lq", default="~/mojolearn-evidence/lq/lq")
    b = sp.add_parser("builds")
    b.add_argument("family", choices=FAMILIES)
    b.add_argument("pairs")
    e = sp.add_parser("extract")
    e.add_argument("board")
    m = sp.add_parser("merge")
    m.add_argument("--logs", required=True)
    m.add_argument("--tsv", default=TSV)
    m.add_argument("--trees-board", default=TREES_BOARD)
    m.add_argument("--out", default=OUT_MD)
    a = ap.parse_args()
    {"select": cmd_select, "builds": cmd_builds, "extract": cmd_extract, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
