#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FSPEED logs from one or more speed legs -> ONE JSON record for the paper.

    python3 tools/identity_grid_json.py <leg-dir> [<leg-dir> ...] --out grid.json

A parser only. It never runs anything and never touches a GPU. It reads the
six FSPEED line kinds (see tools/fast_speed_table.py) from every `*.log`
under each leg's `remote/logs/`, and writes, per (lane, shape, arm):

    median_ms, min_ms, max_ms, rounds, warmup_ms, distinct_hashes,
    metrics {name: value}, mode (from the arm's header), device

and, per (lane, shape), the refusals with their reasons, so a deterministic
arm the vendor does not offer appears with the vendor's own words rather
than as a missing row. The record carries the leg directories, commits and
devices it was built from, so the paper's provenance check can find them.
"""
import argparse
import json
import os
import re
import statistics
import sys

KV = re.compile(r"(\w+)=(\S+)")


def parse_kv(line):
    return dict(KV.findall(line))


def read_leg(leg_dir):
    logs = os.path.join(leg_dir, "remote", "logs")
    if not os.path.isdir(logs):
        logs = leg_dir
    leg = {"dir": leg_dir, "commit": None, "device": None, "ours_mode": None,
           "lines": []}
    lt = os.path.join(leg_dir, "leg.txt")
    if os.path.exists(lt):
        for line in open(lt, encoding="utf-8", errors="replace"):
            if line.startswith("commit="):
                leg["commit"] = line.split("=", 1)[1].strip()
            elif line.startswith("device="):
                leg["device"] = line.split("=", 1)[1].strip()
            elif line.startswith("ours_mode="):
                leg["ours_mode"] = line.split("=", 1)[1].strip()
    for name in sorted(os.listdir(logs)):
        if not name.endswith(".log"):
            continue
        path = os.path.join(logs, name)
        for line in open(path, encoding="utf-8", errors="replace"):
            if line.startswith("FSPEED"):
                leg["lines"].append((name, line.rstrip("\n")))
    return leg


def build(legs):
    cells = {}       # (lane, shape, arm) -> dict
    refusals = {}    # (lane, arm) -> reason (shape-less: refusals carry no shape)
    headers = {}     # (lane, arm) -> {mode, device, rounds}
    notes = []
    for leg in legs:
        for fname, line in leg["lines"]:
            kind, _, rest = line.partition(" ")
            kv = parse_kv(rest)
            lane = kv.get("lane")
            arm = kv.get("arm")
            if kind == "FSPEED-HEADER":
                headers[(lane, arm)] = {"mode": kv.get("mode"),
                                       "device": kv.get("device"),
                                       "rounds": int(kv.get("rounds", 0)),
                                       "log": fname, "leg": leg["dir"]}
            elif kind == "FSPEED":
                key = (lane, kv.get("shape"), arm)
                c = cells.setdefault(key, {"ms": [], "hashes": [], "warmup_ms": None,
                                           "metrics": {}, "log": fname,
                                           "leg": leg["dir"]})
                c["ms"].append(float(kv["ms"]))
                c["hashes"].append(kv.get("hash"))
            elif kind == "FSPEED-WARMUP":
                key = (lane, kv.get("shape"), arm)
                c = cells.setdefault(key, {"ms": [], "hashes": [], "warmup_ms": None,
                                           "metrics": {}, "log": fname,
                                           "leg": leg["dir"]})
                c["warmup_ms"] = float(kv["ms"])
            elif kind == "FSPEED-ACC":
                # ACC lines carry no shape; the log file is one (lane, shape),
                # so attach to every cell of this lane in this file.
                for key, c in cells.items():
                    if key[0] == lane and key[2] == arm and c["log"] == fname:
                        c["metrics"][kv["metric"]] = float(kv["value"])
            elif kind == "FSPEED-REFUSED":
                reason = rest.split("reason=", 1)[1] if "reason=" in rest else rest
                refusals.setdefault((lane, arm, fname), reason)
            elif kind == "FSPEED-NOTE":
                notes.append({"lane": lane, "log": fname, "text": rest})
    out_cells = []
    for (lane, shape, arm), c in sorted(cells.items()):
        h = headers.get((lane, arm), {})
        ms = c["ms"]
        out_cells.append({
            "lane": lane, "shape": shape, "arm": arm,
            "mode": h.get("mode"), "device": h.get("device"),
            "rounds": len(ms),
            "median_ms": statistics.median(ms) if ms else None,
            "min_ms": min(ms) if ms else None,
            "max_ms": max(ms) if ms else None,
            "warmup_ms": c["warmup_ms"],
            "distinct_hashes": len(set(x for x in c["hashes"] if x)),
            "repeats_run_to_run": (len(set(x for x in c["hashes"] if x)) == 1) if ms else None,
            "metrics": c["metrics"],
            "log": c["log"], "leg": c["leg"],
        })
    out_ref = [{"lane": l, "arm": a, "log": f,
                "not_offered": r.startswith("NOT-OFFERED"), "reason": r}
               for (l, a, f), r in sorted(refusals.items())]
    return {"cells": out_cells, "refusals": out_ref, "notes": notes,
            "legs": [{"dir": g["dir"], "commit": g["commit"], "device": g["device"],
                      "ours_mode": g["ours_mode"]} for g in legs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    legs = [read_leg(d) for d in a.legs]
    rec = build(legs)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    print("cells", len(rec["cells"]), "refusals", len(rec["refusals"]),
          "legs", len(rec["legs"]), file=sys.stderr)


if __name__ == "__main__":
    main()
