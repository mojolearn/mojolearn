#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BOARD: every cell of a bench_all_ours.sh sweep, in one table.

    python3 tools/bench_all_summarize.py --out /root/trees_out/bench_all \
        [--tsv board.tsv] [--md board.md]

WHY THIS EXISTS. `tools/bench_all_ours.sh` records WALL SECONDS PER CELL in its
sweep.tsv -- how long the cell took to run, which is a fact about the harness
and not about the library. The measurement itself is in the lines the two
harnesses print: `FSPEED` rounds for the tree lanes, `CTD` records for the
classical ones. Nothing read them back, so a sweep produced a directory of logs
and a human with a grep. This turns those lines into one row per
(lane, dataset, arm): median ms with min..max, the opponent's ms, the ratio,
the quality figure, the output hash, and the fit-equivalence verdict.

WHAT IT WILL NOT DO, AND THIS IS THE POINT
-------------------------------------------
**A CELL THAT DID NOT RUN IS `UNKNOWN`, NEVER A BLANK AND NEVER A ZERO.**
Every lane x dataset x arm the sweep was ASKED for is emitted, whether or not
it produced a number. A missing log is `UNKNOWN(no log)`; a log with no timed
round is `UNKNOWN(no rounds)`; an arm the harness refused carries
`REFUSED(<reason>)`. The failure this guards against is the one the sweep was
commissioned to remove: a board where an absent row reads as a fine row,
because absence is quiet and a number is loud.

Ratios are OURS median / OPPONENT median (below 1 means our median is lower).
A ratio is only computed when BOTH sides produced `--rounds` timed rounds on
THIS pod; a ratio against a cached row from another machine is not computed
here at all, because mixing machines is what this sweep exists to stop.

`FSPEED-FIT-VERDICT` rides on every tree row verbatim. `UNAVAILABLE` and
`UNKNOWN` are carried through as themselves -- never upgraded to COMPARABLE,
never backfilled from a config. Classical rows carry the `CTD-RATIO`
`span_asymmetry=` field for the same reason: ours uploads inside its clock and
the GPU opponents do not, and that runs AGAINST us.
"""
import argparse
import json
import os
import re
import statistics
import sys

TREE_LANES = ("gbdt-symmetric", "gbdt-depthwise", "gbdt-lossguide",
              "rf", "et", "iforest")
CLASSICAL_LANES = ("kmeans", "pca", "ols", "knn", "kde", "svc",
                   "dbscan", "hdbscan")

#: `key=value` where the value runs to the next ` key=` or end of line. The
#: quality field is JSON with spaces in it, so a plain split() loses it.
_KV = re.compile(r"(\S+?)=(.*?)(?=\s+\S+?=|$)")


def kv(line):
    return {m.group(1): m.group(2).strip() for m in _KV.finditer(line.strip())}


def med(xs):
    return float(statistics.median(xs)) if xs else None


def fmt(v, nd=1):
    return "-" if v is None else ("%.*f" % (nd, v) if isinstance(v, float) else str(v))


# ---------------------------------------------------------------------------
# tree lanes: parse the FSPEED contract out of the speed logs
# ---------------------------------------------------------------------------

def parse_tree_log(path):
    """One `trees_identical_ab.sh speed` log -> {arm: record}."""
    arms, verdict, shape = {}, None, None

    def arm(name):
        return arms.setdefault(name, {"ms": [], "hashes": [], "acc": {},
                                      "fit": None, "refused": None})

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if not line.startswith("FSPEED"):
                continue
            head, _, rest = line.partition(" ")
            f = kv(rest)
            name = f.get("arm")
            if head == "FSPEED" and name:
                a = arm(name)
                try:
                    a["ms"].append(float(f["ms"]))
                except (KeyError, ValueError):
                    pass
                if f.get("hash", "-") != "-":
                    a["hashes"].append(f["hash"])
                shape = f.get("shape", shape)
            elif head == "FSPEED-ACC" and name:
                try:
                    arm(name)["acc"][f["metric"]] = float(f["value"])
                except (KeyError, ValueError):
                    pass
            elif head == "FSPEED-FIT" and name:
                arm(name)["fit"] = "trees=%s leaves=%s depth=%s src=%s" % (
                    f.get("trees", "-"), f.get("leaves", "-"),
                    f.get("depth_max", "-"), f.get("source", "-"))
            elif head == "FSPEED-REFUSED" and name:
                arm(name)["refused"] = f.get("reason", "")[:120]
            elif head == "FSPEED-FIT-VERDICT":
                verdict = f.get("verdict", "UNKNOWN")
    return arms, verdict, shape


def tree_rows(out_dir, lanes, datasets, rounds):
    speed = os.path.join(out_dir, "speed")
    rows = []
    for lane in lanes:
        for ds in datasets:
            # trees_identical_ab.sh: <set>.<lane>.<ds>.r<rows>.<mode>[.tag].log
            logs = []
            if os.path.isdir(speed):
                logs = [os.path.join(speed, n) for n in sorted(os.listdir(speed))
                        if n.endswith(".log")
                        and ".%s.%s.r" % (lane, ds) in "." + n]
            if not logs:
                rows.append(dict(lane=lane, dataset=ds, arm="ours", shape="-",
                                 status="UNKNOWN(no log)"))
                continue
            arms, verdict, shape = parse_tree_log(logs[-1])
            if not arms:
                rows.append(dict(lane=lane, dataset=ds, arm="ours", shape=shape or "-",
                                 status="UNKNOWN(no FSPEED lines)", log=logs[-1]))
                continue
            ours_med = med(arms.get("ours", {}).get("ms", []))
            for name in sorted(arms):
                a = arms[name]
                ok = len(a["ms"]) >= rounds
                m = med(a["ms"]) if a["ms"] else None
                if a["refused"] and not a["ms"]:
                    status = "REFUSED(%s)" % a["refused"]
                elif not a["ms"]:
                    status = "UNKNOWN(no rounds)"
                elif not ok:
                    status = "PARTIAL(%d/%d rounds)" % (len(a["ms"]), rounds)
                else:
                    status = "ok"
                ratio = None
                if name != "ours" and ours_med and m and ok:
                    ratio = ours_med / m
                rows.append(dict(
                    lane=lane, dataset=ds, arm=name, shape=shape or "-",
                    status=status, median_ms=m,
                    min_ms=min(a["ms"]) if a["ms"] else None,
                    max_ms=max(a["ms"]) if a["ms"] else None,
                    rounds=len(a["ms"]),
                    quality=",".join("%s=%.6g" % kvp for kvp in sorted(a["acc"].items())) or "-",
                    hash=(a["hashes"][-1] if a["hashes"] else "-"),
                    hash_stable=(len(set(a["hashes"])) == 1) if a["hashes"] else None,
                    fit=a["fit"] or "UNAVAILABLE",
                    fit_verdict=verdict or "UNKNOWN",
                    ratio_ours_over=ratio, log=logs[-1]))
    return rows


# ---------------------------------------------------------------------------
# classical lanes: the race already writes the whole record as JSON
# ---------------------------------------------------------------------------

def classical_rows(out_dir, lanes, datasets, rounds):
    rows = []
    for lane in lanes:
        for ds in datasets:
            path = None
            for cand in (os.path.join(out_dir, "ctd", "%s-%s.json" % (lane, ds)),
                         os.path.join(out_dir, "ctd-%s-%s" % (lane, ds),
                                      "%s-%s.json" % (lane, ds))):
                if os.path.exists(cand):
                    path = cand
                    break
            if path is None:
                rows.append(dict(lane=lane, dataset=ds, arm="ours", shape="-",
                                 status="UNKNOWN(no race json)"))
                continue
            with open(path) as fh:
                r = json.load(fh)
            shapes = r.get("block", {}).get("arrays", {})
            first = shapes.get("X") or shapes.get("index") or {}
            shape = "x".join(str(s) for s in first.get("shape", [])) or "-"
            qual = r.get("quality", {}) or {}
            ours = r.get("arms", {}).get("ours", {})
            ours_med = ours.get("median_ms")
            for name in sorted(r.get("arms", {})):
                a = r["arms"][name]
                ms = a.get("ms") or []
                if a.get("status") != "ok" and not ms:
                    status = "REFUSED(%s)" % str(a.get("status"))[:60]
                elif len(ms) < rounds:
                    status = "PARTIAL(%d/%d rounds)" % (len(ms), rounds)
                else:
                    status = "ok"
                q = qual.get(name) or {}
                ratio = (r.get("ratios_ours_over", {}) or {}).get(name)
                span = a.get("span", {}) or {}
                rows.append(dict(
                    lane=lane, dataset=ds, arm=name, shape=shape, status=status,
                    median_ms=a.get("median_ms"), min_ms=a.get("min_ms"),
                    max_ms=a.get("max_ms"), rounds=len(ms),
                    quality=",".join(
                        "%s=%s" % (k, ("%.6g" % v) if isinstance(v, float) else v)
                        for k, v in sorted(q.items()) if k != "reference") or "-",
                    hash=(a.get("digests") or ["-"])[-1],
                    hash_stable=a.get("digest_stable"),
                    fit="n/a (classical)",
                    fit_verdict="n/a",
                    span="input_home=%s pre_clock_fit=%s" % (
                        span.get("input_home", "-"), span.get("pre_clock_fit", "-")),
                    ratio_ours_over=(None if name == "ours" else ratio),
                    log=path))
    return rows


# ---------------------------------------------------------------------------

COLUMNS = ("lane", "dataset", "shape", "arm", "status", "median_ms", "min_ms",
           "max_ms", "rounds", "ratio_ours_over", "quality", "hash",
           "hash_stable", "fit", "fit_verdict", "span")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True,
                    help="the sweep's output dir (holds speed/ and ctd/)")
    ap.add_argument("--rounds", type=int, default=5,
                    help="rounds the sweep asked for; a cell with fewer is PARTIAL")
    ap.add_argument("--lanes", default=",".join(TREE_LANES + CLASSICAL_LANES))
    ap.add_argument("--datasets", default="taxi,istella")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    lanes = [l for l in args.lanes.split(",") if l]
    datasets = [d for d in args.datasets.split(",") if d]
    rows = tree_rows(args.out, [l for l in lanes if l in TREE_LANES],
                     datasets, args.rounds)
    rows += classical_rows(args.out, [l for l in lanes if l in CLASSICAL_LANES],
                           datasets, args.rounds)

    out = []
    out.append("\t".join(COLUMNS))
    for r in rows:
        out.append("\t".join(
            fmt(r.get(c), 3 if c == "ratio_ours_over" else 1)
            if isinstance(r.get(c), float) else str(r.get(c, "-"))
            for c in COLUMNS))
    text = "\n".join(out) + "\n"
    sys.stdout.write(text)
    if args.tsv:
        with open(args.tsv, "w") as fh:
            fh.write(text)

    unknown = [r for r in rows if str(r.get("status", "")).startswith("UNKNOWN")]
    refused = [r for r in rows if str(r.get("status", "")).startswith("REFUSED")]
    partial = [r for r in rows if str(r.get("status", "")).startswith("PARTIAL")]
    sys.stdout.write("\nCELLS: %d total, %d ok, %d UNKNOWN, %d REFUSED, %d PARTIAL\n"
                     % (len(rows), len(rows) - len(unknown) - len(refused) - len(partial),
                        len(unknown), len(refused), len(partial)))
    for r in unknown + partial:
        sys.stdout.write("  %-16s %-8s %-18s %s\n"
                         % (r["lane"], r["dataset"], r.get("arm", "-"), r["status"]))

    if args.md:
        with open(args.md, "w") as fh:
            fh.write("| " + " | ".join(COLUMNS) + " |\n")
            fh.write("|" + "---|" * len(COLUMNS) + "\n")
            for r in rows:
                fh.write("| " + " | ".join(
                    fmt(r.get(c), 3 if c == "ratio_ours_over" else 1)
                    if isinstance(r.get(c), float) else str(r.get(c, "-"))
                    for c in COLUMNS) + " |\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
