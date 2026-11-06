#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Per-lane timing of two sets of identity records, before against after.

    python3 tools/identity_timing_compare.py --before <record|dir>... --after <record|dir>...
        [--min-ratio 1.5] [--min-seconds 0.5] [--top N] [--json]

Reads what the records already store (`cells[<lane>/<fixture>].timing`, written
by `CellTimer` in tools/identity_break.py); it runs nothing and changes nothing.

* Each side is a list of identity records, oldest first. When several records
  of one side hold the same cell (a reuse record that re-recorded only the
  changed lanes, a shard, a resumed record), the record given LAST supplies it,
  as in tools/identity_lane_timings.py.
* Only `repeat=1/<stage>` stages are summed, so a record taken with repeats=2
  compares with one taken with repeats=1.
* Stages are bucketed: `train`, `infer` (infer-save-reload) and `other`
  (batch, batchgrad, batchscale, ragged, stepfull, rlpair and anything else).
  A REFUSED cell, or one carrying an `error`, stopped early and is not timed.
  A lane loads its binding inside its first fit, so binding load sits in train.
* A lane is compared over the fixtures timed on BOTH sides only. It is SKIPPED,
  and listed, when the supplying records carry different `lane_revisions` for it
  (its fixture, vocabulary or arithmetic moved, tools/identity_break.py
  LANE_REVISIONS) or when a common fixture's data hash differs. A lane whose
  `batch_revisions` entry differs is compared and marked `batch-rev`.
* Each side lists the vendors (device labels) it read; mixing hardware on one
  side is reported, never hidden.

Timings are informational: no digest, closure, reuse key or admission reads them.
"""
import argparse
import json
import sys
from pathlib import Path

BUCKETS = ("train", "infer", "other")


def _files(paths):
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*.json") if p.is_file())
        else:
            yield path


def bucket(stage):
    """`repeat=1/train` -> train; None for another repeat."""
    prefix, _, name = stage.rpartition("/")
    if prefix and prefix != "repeat=1":
        return None
    if name == "train":
        return "train"
    if name.startswith("infer"):
        return "infer"
    return "other"


def cell_seconds(timing):
    """{bucket: seconds} from one cell's timing, repeat 1 only; None if untimed."""
    if not (isinstance(timing, dict) and isinstance(timing.get("total_seconds"), (int, float))):
        return None
    out = dict.fromkeys(BUCKETS, 0.0)
    stages = timing.get("stages") or {}
    if not stages:
        out["train"] = float(timing["total_seconds"])
        return out
    for name, seconds in stages.items():
        b = bucket(str(name))
        if b is not None and isinstance(seconds, (int, float)):
            out[b] += float(seconds)
    return out


def ran(cell):
    """A cell that did its work: a REFUSED cell (binding not built, worker
    failed) or one carrying an `error` stops early, so its time is not the
    lane's time and is never compared."""
    return str(cell.get("verdict") or "").upper() != "REFUSED" and "error" not in cell


def load_side(paths):
    """{(lane, fixture): dict(sec, rev, brev, fx, src)} plus the records read."""
    cells, records = {}, []
    for path in _files(paths):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not (isinstance(data, dict) and isinstance(data.get("cells"), dict) and "fixtures" in data):
            continue
        revs = data.get("lane_revisions") or {}
        brevs = data.get("batch_revisions") or {}
        fixtures = data.get("fixtures") or {}
        meta = dict(path=str(path), vendor=str(data.get("vendor")), commit=str(data.get("commit") or ""),
                    repeats=data.get("repeats"), timed=0)
        for key, cell in data["cells"].items():
            lane, _, fixture = key.partition("/")
            if not isinstance(cell, dict) or not ran(cell):
                meta["not_run"] = meta.get("not_run", 0) + 1
                continue
            sec = cell_seconds(cell.get("timing"))
            if sec is None:
                continue
            meta["timed"] += 1
            fx = fixtures.get(fixture)
            cells[(lane, fixture)] = dict(sec=sec, rev=revs.get(lane), brev=brevs.get(lane),
                                          fx=json.dumps(fx, sort_keys=True) if fx is not None else None,
                                          src=str(path), commit=meta["commit"])
        records.append(meta)
    return cells, records


def compare(before_paths, after_paths, min_ratio=1.5, min_seconds=0.5):
    before, brecs = load_side(before_paths)
    after, arecs = load_side(after_paths)
    lanes = {}
    skipped = {}
    for (lane, fixture), b in before.items():
        a = after.get((lane, fixture))
        if a is None:
            continue
        if b["rev"] != a["rev"]:
            skipped[lane] = f"lane revision {b['rev']} -> {a['rev']}"
            continue
        if b["fx"] is not None and a["fx"] is not None and b["fx"] != a["fx"]:
            skipped[lane] = f"fixture {fixture} data changed"
            continue
        e = lanes.setdefault(lane, dict(lane=lane, fixtures=0, batch_rev_changed=False,
                                        before=dict.fromkeys(BUCKETS, 0.0), after=dict.fromkeys(BUCKETS, 0.0),
                                        before_commits=set(), after_commits=set()))
        e["fixtures"] += 1
        e["batch_rev_changed"] |= b["brev"] != a["brev"]
        e["before_commits"].add(b["commit"][:10])
        e["after_commits"].add(a["commit"][:10])
        for k in BUCKETS:
            e["before"][k] += b["sec"][k]
            e["after"][k] += a["sec"][k]
    for lane in skipped:
        lanes.pop(lane, None)
    rows = []
    for e in lanes.values():
        bt, at = sum(e["before"].values()), sum(e["after"].values())
        deltas = {k: e["after"][k] - e["before"][k] for k in BUCKETS}
        grew = max(BUCKETS, key=lambda k: deltas[k])
        e.update(before_s=bt, after_s=at, delta_s=at - bt,
                 ratio=(at / bt) if bt > 0 else None, grew=grew if deltas[grew] > 0 else "-",
                 regressed=bool(bt > 0 and at / bt >= min_ratio and at - bt >= min_seconds),
                 improved=bool(at > 0 and bt / at >= min_ratio and bt - at >= min_seconds),
                 before_commits=sorted(e["before_commits"]), after_commits=sorted(e["after_commits"]))
        rows.append(e)
    rows.sort(key=lambda r: (-r["delta_s"], r["lane"]))
    # The host factor: the median after/before ratio over lanes that took at
    # least 0.2 s before. Two pods of one GPU model differ (host CPU, driver,
    # cold caches), and that moves every lane together; a lane far above the
    # median moved for its own reason. `rel` = ratio / host factor.
    steady = sorted(r["ratio"] for r in rows if r["ratio"] is not None and r["before_s"] >= 0.2)
    factor = steady[len(steady) // 2] if steady else None
    for r in rows:
        r["rel"] = (r["ratio"] / factor) if (factor and r["ratio"] is not None) else None
        r["outlier"] = bool(r["regressed"] and r["rel"] is not None and r["rel"] >= min_ratio)
    totals = {side: {k: sum(r[side][k] for r in rows) for k in BUCKETS} for side in ("before", "after")}
    before_lanes = {lane for lane, _ in before}
    after_lanes = {lane for lane, _ in after}
    return dict(
        before_records=brecs, after_records=arecs,
        before_vendors=sorted({r["vendor"] for r in brecs if r["timed"]}),
        after_vendors=sorted({r["vendor"] for r in arecs if r["timed"]}),
        rows=rows, totals=totals, host_factor=factor,
        skipped=dict(sorted(skipped.items())),
        only_before=sorted(before_lanes - after_lanes - set(skipped)),
        only_after=sorted(after_lanes - before_lanes - set(skipped)),
        regressed=[r["lane"] for r in rows if r["regressed"]],
        outliers=[r["lane"] for r in rows if r["outlier"]],
        improved=[r["lane"] for r in rows if r["improved"]],
    )


def _fmt(x):
    return "-" if x is None else f"{x:.2f}"


def _print_text(rep, top):
    print(f"before vendors: {', '.join(rep['before_vendors']) or '-'}; after vendors: {', '.join(rep['after_vendors']) or '-'}")
    if len(rep["before_vendors"]) > 1 or len(rep["after_vendors"]) > 1:
        print("WARNING: one side mixes device labels")
    print(f"host factor (median ratio, lanes >= 0.2 s before): {_fmt(rep['host_factor'])}")
    print(f"{'lane':<40} {'before_s':>9} {'after_s':>9} {'ratio':>7} {'rel':>6} {'delta_s':>8} grew   flag")
    for r in rep["rows"][:top]:
        flag = "REGRESSED" if r["regressed"] else ("faster" if r["improved"] else "")
        if r["outlier"]:
            flag += " outlier"
        if r["batch_rev_changed"]:
            flag += " batch-rev"
        print(f"{r['lane']:<40} {r['before_s']:>9.2f} {r['after_s']:>9.2f} {_fmt(r['ratio']):>7} "
              f"{r['delta_s']:>8.2f} {r['grew']:<6} {flag}")
    for side in ("before", "after"):
        t = rep["totals"][side]
        print(f"TOTAL {side}: " + " ".join(f"{k}={t[k]:.1f}" for k in BUCKETS) + f" all={sum(t.values()):.1f} s")
    print(f"lanes compared {len(rep['rows'])}; regressed {len(rep['regressed'])} ({len(rep['outliers'])} above the host factor too); improved {len(rep['improved'])}; "
          f"skipped (revision/fixture) {len(rep['skipped'])}; only before {len(rep['only_before'])}; "
          f"only after {len(rep['only_after'])}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", nargs="+", required=True, help="records or dirs, oldest first")
    parser.add_argument("--after", nargs="+", required=True, help="records or dirs, oldest first")
    parser.add_argument("--min-ratio", type=float, default=1.5)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rep = compare(args.before, args.after, args.min_ratio, args.min_seconds)
    if args.json:
        json.dump(rep, sys.stdout, indent=1)
        print()
    else:
        _print_text(rep, args.top)
    return 0 if rep["before_records"] and rep["after_records"] else 1


if __name__ == "__main__":
    sys.exit(main())
