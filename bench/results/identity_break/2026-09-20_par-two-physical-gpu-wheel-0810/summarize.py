#!/usr/bin/env python3
"""summarize.py <vendor dir>: count what the verifier's own JSON documents say.

Reads every `lease*/par_all_<fixture>.out` (format mojolearn.verify-par.v1)
under the vendor directory and prints, without reinterpreting any verdict:
lanes, fixtures, parts per verdict, a lane x verdict table, every part that
is not IDENTICAL or N/A with the verifier's own hashes and error text, and
the placement witness (distinct physical GPU UUIDs per cell). It also lists
the exit code of every command from each lease's status.tsv.

If a fixture was run in more than one lease, every run is counted and the
duplication is reported; nothing is silently dropped.
"""
import glob
import json
import os
import sys

root = sys.argv[1]
docs = []
for path in sorted(glob.glob(os.path.join(root, "lease*", "par_*.out"))):
    name = os.path.basename(path)[:-4]
    if name.startswith("par_self_test"):
        continue
    try:
        doc = json.load(open(path))
    except Exception as exc:  # a timed-out command leaves no document
        print(f"NO DOCUMENT: {os.path.relpath(path, root)} ({type(exc).__name__})")
        continue
    docs.append((os.path.relpath(path, root), name, doc))

print("## commands (status.tsv: name, exit, seconds)")
for st in sorted(glob.glob(os.path.join(root, "lease*", "status.tsv"))):
    print(os.path.relpath(st, root))
    for line in open(st):
        print("   ", line.rstrip("\n").replace("\t", "  "))

def table(selected, title):
    print(f"\n## {title}")
    lanes, fixtures, counts, per_lane, seen = set(), [], {}, {}, {}
    uuids, cells_witnessed, cells_total, refusals = {}, 0, 0, []
    for rel, name, doc in selected:
        if doc.get("ran") is False:
            print(f"{rel}: NOT RUN: {doc.get('reason')}")
            continue
        print(f"{rel}: state={doc.get('state')} passed={doc.get('passed')} vendor={doc.get('vendor')} "
              f"devices={doc.get('devices')} repeats={doc.get('repeats')} lanes={len(doc.get('lanes', []))} "
              f"fixtures={doc.get('fixtures')} parts={doc.get('compared')} counts={doc.get('counts')} "
              f"elapsed_s={doc.get('elapsed_s')} witness_refusal={doc.get('witness_refusal')}")
        for fx in doc.get("fixtures", []):
            if fx not in fixtures:
                fixtures.append(fx)
        for c in doc.get("cells", []):
            lanes.add(c["lane"])
            counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
            per_lane.setdefault(c["lane"], {}).setdefault(c["verdict"], 0)
            per_lane[c["lane"]][c["verdict"]] += 1
        for w in doc.get("cell_witnesses", []):
            cells_total += 1
            if w.get("witness_refusal"):
                refusals.append((w["lane"], w["fixture"], w["witness_refusal"]))
            seen[(w["lane"], w["fixture"])] = seen.get((w["lane"], w["fixture"]), 0) + 1
            cell_uuids = set()
            for pool in w["witness"].get("detail", []):
                for worker in pool.get("workers", []):
                    cell_uuids.update(u for u in worker.get("devices", []) if u)
            if len(cell_uuids) >= 2:
                cells_witnessed += 1
            uuids.setdefault(rel.split(os.sep)[0], set()).update(cell_uuids)
    dup = {f"{l}/{f}": n for (l, f), n in seen.items() if n > 1}
    print(f"\nlane/fixture cells run: {len(seen)} (59 lanes x 9 fixtures = 531 is the whole of `--par all`)")
    total = sum(counts.values())
    compared = sum(v for k, v in counts.items() if k in ("IDENTICAL", "DIVERGENT", "MOVED"))
    print(f"lanes run: {len(lanes)}   fixtures run: {len(fixtures)} {fixtures}")
    if dup:
        print(f"CELLS RUN MORE THAN ONCE (every run is counted): {dup}")
    print(f"parts reported: {total}   parts with two hashes compared: {compared}")
    for k in ("IDENTICAL", "DIVERGENT", "MOVED", "ONE-COLUMN", "REFUSED", "CPU-ROUTE-LIMIT", "N/A"):
        print(f"  {k:<16} {counts.get(k, 0)}")
    other = {k: v for k, v in counts.items()
             if k not in ("IDENTICAL", "DIVERGENT", "MOVED", "ONE-COLUMN", "REFUSED", "CPU-ROUTE-LIMIT", "N/A")}
    if other:
        print(f"  OTHER VERDICTS: {other}")
    print(f"placement witness: {cells_witnessed} of {cells_total} lane/fixture cells showed >= 2 distinct "
          f"physical GPU UUIDs; witness refusals: {len(refusals)}")
    for lease in sorted(uuids):
        print(f"  {lease}: GPU UUIDs named by the workers themselves: {sorted(uuids[lease])}")
    for lane, fx, why in refusals:
        print(f"  WITNESS REFUSED {lane}/{fx}: {why}")
    cols = ("IDENTICAL", "DIVERGENT", "MOVED", "ONE-COLUMN", "REFUSED", "N/A")
    refused_by_lane = {}
    for lane, fx, why in refusals:
        refused_by_lane[lane] = refused_by_lane.get(lane, 0) + 1
    cells_by_lane = {}
    for (lane, fx), n in seen.items():
        cells_by_lane[lane] = cells_by_lane.get(lane, 0) + n
    print("\n| lane | fixtures run | " + " | ".join(cols) + " | WITNESS REFUSED (fixtures) |")
    print("|---|---|" + "---|" * len(cols) + "---|")
    for lane in sorted(per_lane):
        print(f"| {lane} | {cells_by_lane.get(lane, 0)} | "
              + " | ".join(str(per_lane[lane].get(k, 0)) for k in cols)
              + f" | {refused_by_lane.get(lane, 0)} |")
    print("\nevery part that is not IDENTICAL and not N/A, in the verifier's own words:")
    n = 0
    for rel, name, doc in selected:
        for c in doc.get("cells", []):
            if c["verdict"] in ("IDENTICAL", "N/A"):
                continue
            n += 1
            print(f"  {c['lane']} {c['fixture']} {c['part']} {c['verdict']} one={c['one']} two={c['two']}")
            for side in ("one", "two"):
                if c.get(f"{side}_error"):
                    print(f"      {side}-device error: {str(c[f'{side}_error'])[-600:]}")
    if not n:
        print("  (none)")

table([d for d in docs if d[1] == "par_quick"], "verify --par quick")
table([d for d in docs if d[1].startswith("par_all_")], "verify --par all (one fixture per command)")
if any(d[1].startswith("par_lanes_") for d in docs):
    table([d for d in docs if d[1].startswith("par_lanes_")],
          "verify --par --lanes <explicit list> --fixtures base")
