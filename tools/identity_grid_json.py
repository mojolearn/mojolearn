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
           "lanes": None, "pod": None, "started": None, "finished": None,
           "lightgbm_cuda": None, "lines": []}
    # The local leg.txt carries the rental (pod, started, finished) and an
    # abbreviated `commit=<short> parent <short>`; the box's remote/leg.txt
    # carries the 40-hex commit it was built from, the mode our arm was
    # compiled in (DEVIATION 1898) and the device it read back. The remote
    # record wins for the source facts; the local one for the rental facts.
    for rel in ("leg.txt", os.path.join("remote", "leg.txt")):
        lt = os.path.join(leg_dir, rel)
        if not os.path.exists(lt):
            continue
        for line in open(lt, encoding="utf-8", errors="replace"):
            k, eq, v = line.rstrip("\n").partition("=")
            if not eq:
                continue
            v = v.strip()
            if k == "commit" and re.fullmatch(r"[0-9a-f]{40}", v):
                leg["commit"] = v
            elif k == "commit" and leg["commit"] is None:
                leg["commit"] = v.split()[0]
            elif k in ("device", "ours_mode", "lanes", "pod", "started", "finished"):
                if k in ("started", "finished") and rel != "leg.txt":
                    continue
                leg[k] = v
            elif k in ("lightgbm_cuda_build", "lightgbm_cuda_works", "lightgbm_cuda_build_exit"):
                leg["lightgbm_cuda"] = ((leg["lightgbm_cuda"] + "; ") if leg["lightgbm_cuda"] else "") + k + "=" + v
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
    acc = []
    for leg in legs:
        for fname, line in leg["lines"]:
            kind, _, rest = line.partition(" ")
            kv = parse_kv(rest)
            lane = kv.get("lane")
            arm = kv.get("arm")
            # DEVIATION 2212: the compiled classical driver prints arm=ours;
            # when the leg ran it beside the Python-API ours arm it wrote the
            # driver's log as `*.ours-native.log`, and that is the arm name.
            if arm == "ours" and fname.endswith(".ours-native.log"):
                arm = "ours-native"
            if kind == "FSPEED-HEADER":
                headers[(lane, arm)] = {"mode": kv.get("mode"),
                                       "device": kv.get("device"),
                                       "family": kv.get("family"),
                                       "size": kv.get("size"),
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
                # ACC lines carry no shape and may precede the first round
                # (an initial-loss line); they are attached to every cell of
                # this (lane, arm) in this file after the file is read.
                acc.append((lane, arm, fname, kv["metric"], float(kv["value"])))
            elif kind == "FSPEED-REFUSED":
                reason = rest.split("reason=", 1)[1] if "reason=" in rest else rest
                refusals.setdefault((lane, arm, fname), reason)
            elif kind == "FSPEED-NOTE":
                notes.append({"lane": lane, "log": fname, "text": rest})
    for lane, arm, fname, metric, value in acc:
        for key, c in cells.items():
            if key[0] == lane and key[2] == arm and c["log"] == fname:
                c["metrics"][metric] = value
    out_cells = []
    for (lane, shape, arm), c in sorted(cells.items()):
        h = headers.get((lane, arm), {})
        ms = c["ms"]
        if not ms:
            # A warm-up with no timed round is not a measurement; it is
            # recorded as a note so the absence is visible, not silent.
            notes.append({"lane": lane, "log": c["log"],
                          "text": "arm=%s shape=%s warm-up only (%s ms), no timed round"
                                  % (arm, shape, c["warmup_ms"])})
            continue
        # The FSPEED-HEADER `mode=` field is the HARNESS PROCESS's numeric
        # mode label (tools/speed_gbdt_arm.py::numeric_mode_label), which is
        # only a statement about OUR arm. A vendor arm's configuration is
        # what its name says: the plain name is the vendor's default (its
        # fast configuration) and the `-deterministic` sibling is the
        # vendor's documented deterministic configuration.
        if arm == "ours":
            mode = h.get("mode")
        elif "-deterministic" in arm:
            mode = "VENDOR-DETERMINISTIC"
        else:
            mode = "VENDOR-DEFAULT"
        out_cells.append({
            "lane": lane, "shape": shape, "arm": arm,
            "family": h.get("family"), "size": h.get("size"),
            "mode": mode, "device": h.get("device"),
            "rounds": len(ms),
            "median_ms": statistics.median(ms) if ms else None,
            "min_ms": min(ms) if ms else None,
            "max_ms": max(ms) if ms else None,
            "warmup_ms": c["warmup_ms"],
            "distinct_hashes": len(set(x for x in c["hashes"] if x)),
            # One round cannot witness repetition; the field is None below two.
            "repeats_run_to_run": (len(set(x for x in c["hashes"] if x)) == 1) if len(ms) >= 2 else None,
            "metrics": c["metrics"],
            "log": c["log"], "leg": c["leg"],
        })
    out_ref = [{"lane": l, "arm": a, "log": f,
                "not_offered": r.startswith("NOT-OFFERED"), "reason": r}
               for (l, a, f), r in sorted(refusals.items())]
    return {"cells": out_cells, "refusals": out_ref, "notes": notes,
            "legs": [{k: g[k] for k in ("dir", "commit", "device", "ours_mode", "lanes",
                                        "pod", "started", "finished", "lightgbm_cuda")}
                     for g in legs]}


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
