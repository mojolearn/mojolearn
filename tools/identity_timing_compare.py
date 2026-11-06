#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Per-lane timing of identity records: one column before against after, several
columns side by side, or one lane's series over commits.

    python3 tools/identity_timing_compare.py --before <record|dir>... --after <record|dir>...
        [--min-ratio 1.5] [--min-seconds 0.5] [--top N] [--json]
    python3 tools/identity_timing_compare.py cross --before COL=<path>[,<path>...] ... \\
        --after COL=<path>[,<path>...] ... [--ref COL] [--top N] [--json]
    python3 tools/identity_timing_compare.py series --lane LANE [--lane LANE...] \\
        [--stage non-train|all|train|infer|other] [--commits C1,C2,...] [--common-only] <record|dir>...

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
* Every table header names the commit (and its date) each side read, and the
  machine (vendor label, CPU model). Mixing hardware or commits on one side is
  reported, never hidden.

THE PAIRING RULE (2026-11-06). A statement that compares columns (the CPU
column regressed more than AMD; NVIDIA pays three times the per-call cost) is
only made when the columns were recorded at the SAME commit on the side the
statement is about. `cross` checks it per side: when every column's after side
(and before side) read the same commit set it prints COMPARABLE and the
column-against-reference ratios; otherwise it prints the commits side by side,
marks the statement NOT COMPARABLE and prints no cross-column ratio. The
2026-11-05 regression report broke this rule: it set the CPU column at Oct 25
(b25df4e213) beside the GPU columns at Nov 5 and blamed the CPU column for a
per-call cost every column paid at b25df4e213. `series` prints one lane per
commit across all columns, which is how the same-commit picture is read.

Timings are informational: no digest, closure, reuse key or admission reads them.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

BUCKETS = ("train", "infer", "other")
STAGES = {
    "all": BUCKETS,
    "non-train": ("infer", "other"),
    "train": ("train",),
    "infer": ("infer",),
    "other": ("other",),
}
ROOT = Path(__file__).resolve().parent.parent

_DATES = {}


def _files(paths):
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*.json") if p.is_file())
        else:
            yield path


def column_of(vendor):
    """The column a vendor label belongs to: `nvidia-ptx-*` -> ptx,
    `nvidia-*`/`amd-*`/`apple-*`/`cpu-*` -> that word; anything else as is."""
    v = str(vendor or "")
    if v.startswith("nvidia-ptx"):
        return "ptx"
    head = v.split("-", 1)[0]
    return head if head in ("nvidia", "amd", "apple", "cpu") else v


def commit_date(commit, repo=None):
    """`YYYY-MM-DD` committer date of a commit from git, cached; '' when the
    commit is unknown to the repository (or git is missing)."""
    commit = str(commit or "")
    if not commit:
        return ""
    if commit in _DATES:
        return _DATES[commit]
    try:
        out = subprocess.run(["git", "-C", str(repo or ROOT), "show", "-s", "--format=%cs", commit],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    _DATES[commit] = out.splitlines()[0] if out else ""
    return _DATES[commit]


def load_commit_dates(path):
    """Seed the date cache from a `<commit>\\t<YYYY-MM-DD>` file (prefixes allowed)."""
    for line in Path(path).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] and parts[1]:
            _DATES[parts[0]] = parts[1][:10]


def _date_of(commit):
    """Date by full commit, else by any cached prefix of it, else from git."""
    if commit in _DATES:
        return _DATES[commit]
    for key, value in _DATES.items():
        if key and commit.startswith(key):
            return value
    return commit_date(commit)


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


def record_host(data):
    """What the wall clock ran on, from the record: the `timing_env` object
    (identity_break.py since 2026-11-06) or, for older records, the CPU
    column's `host.cpu_model`. Fields a record does not state are None."""
    env = data.get("timing_env") if isinstance(data.get("timing_env"), dict) else {}
    host = data.get("host") if isinstance(data.get("host"), dict) else {}
    threads = env.get("threads") if isinstance(env.get("threads"), dict) else {}
    return dict(
        cpu_model=env.get("cpu_model") or host.get("cpu_model"),
        logical_cpus=env.get("logical_cpus"), affinity_cpus=env.get("affinity_cpus"),
        cgroup_cpu_quota=env.get("cgroup_cpu_quota"),
        cpu_threads=threads.get("MOJOLEARN_CPU_THREADS"),
        threads={k: v for k, v in threads.items() if v is not None} if threads else None,
        concurrency=env.get("concurrency"), parts=data.get("timing_env_parts"),
        stated=bool(env),
    )


def hosts_comparable(a, b):
    """(True, []) when two record hosts may be timed against each other;
    otherwise (False, [differences]). A field neither record states is not a
    difference; a field one states and the other does not is listed as
    unknown, and only a stated, differing value makes them not comparable."""
    diffs, unknown = [], []
    for key in ("cpu_model", "logical_cpus", "affinity_cpus", "cgroup_cpu_quota", "threads", "concurrency"):
        va, vb = a.get(key), b.get(key)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            unknown.append(f"{key}: {va!r} vs {vb!r} (one unrecorded)")
        elif va != vb:
            diffs.append(f"{key}: {va!r} vs {vb!r}")
    return (not diffs), diffs + unknown


def load_side(paths):
    """{(lane, fixture): dict(sec, rev, brev, fx, src, commit)} plus the records read."""
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
        meta = dict(path=str(path), vendor=str(data.get("vendor")), column=column_of(data.get("vendor")),
                    commit=str(data.get("commit") or ""), repeats=data.get("repeats"), timed=0,
                    host=record_host(data))
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


def side_points(records, cells=None):
    """The commits one side read, oldest record first: [{commit, date, vendors,
    cpu_models, records, cells}]. With `cells`, `cells` counts the cells each
    commit SUPPLIED (after last-wins), else the cells its records timed."""
    supplied = {}
    if cells is not None:
        for c in cells.values():
            supplied[c["commit"]] = supplied.get(c["commit"], 0) + 1
    out = {}
    for r in records:
        if not r["timed"]:
            continue
        p = out.setdefault(r["commit"], dict(commit=r["commit"][:10], full=r["commit"], date=_date_of(r["commit"]),
                                             vendors=set(), cpu_models=set(), records=0, cells=0))
        p["vendors"].add(r["vendor"])
        if r["host"].get("cpu_model"):
            p["cpu_models"].add(r["host"]["cpu_model"])
        p["records"] += 1
        p["cells"] += r["timed"]
    for full, p in out.items():
        if cells is not None:
            p["cells"] = supplied.get(full, 0)
        p["vendors"] = sorted(p["vendors"])
        p["cpu_models"] = sorted(p["cpu_models"])
    return [p for p in out.values() if cells is None or p["cells"]]


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
    bpts, apts = side_points(brecs, before), side_points(arecs, after)
    bhost = _side_host(brecs)
    ahost = _side_host(arecs)
    same_host, host_diffs = hosts_comparable(bhost, ahost)
    bvend = sorted({r["vendor"] for r in brecs if r["timed"]})
    avend = sorted({r["vendor"] for r in arecs if r["timed"]})
    if bvend and avend and bvend != avend:
        # A changed device label is a changed machine (MI325X -> MI300X).
        same_host = False
        host_diffs = [f"vendor: {','.join(bvend)} vs {','.join(avend)}"] + host_diffs
    return dict(
        before_records=brecs, after_records=arecs,
        before_vendors=sorted({r["vendor"] for r in brecs if r["timed"]}),
        after_vendors=sorted({r["vendor"] for r in arecs if r["timed"]}),
        before_points=bpts, after_points=apts,
        before_host=bhost, after_host=ahost, same_host=same_host, host_differences=host_diffs,
        rows=rows, totals=totals, host_factor=factor,
        skipped=dict(sorted(skipped.items())),
        only_before=sorted(before_lanes - after_lanes - set(skipped)),
        only_after=sorted(after_lanes - before_lanes - set(skipped)),
        regressed=[r["lane"] for r in rows if r["regressed"]],
        outliers=[r["lane"] for r in rows if r["outlier"]],
        improved=[r["lane"] for r in rows if r["improved"]],
    )


def _side_host(records):
    """The host of a side: the last timed record's (the one that supplies
    most cells); `mixed` lists every distinct CPU model when they differ."""
    timed = [r for r in records if r["timed"]]
    if not timed:
        return {}
    host = dict(timed[-1]["host"])
    models = sorted({r["host"]["cpu_model"] for r in timed if r["host"].get("cpu_model")})
    if len(models) > 1:
        host["mixed_cpu_models"] = models
    return host


def commit_set(points):
    return tuple(sorted(p["full"] for p in points))


def pairing(columns):
    """THE PAIRING RULE. `columns` = {name: report from compare()}. Per side,
    `same` is True when every column read the same set of commits. A
    cross-column statement is COMPARABLE only when both sides are the same;
    `after_only` marks a set whose after sides agree but whose before sides
    differ (a statement about the after state alone stays comparable)."""
    out = dict(columns={}, before_same=False, after_same=False)
    for side in ("before", "after"):
        sets = {name: commit_set(rep[f"{side}_points"]) for name, rep in columns.items()}
        nonempty = [s for s in sets.values() if s]
        out[f"{side}_same"] = bool(nonempty) and len(nonempty) == len(sets) and len(set(nonempty)) == 1
    for name, rep in columns.items():
        out["columns"][name] = {side: [dict(commit=p["commit"], date=p["date"], cells=p["cells"])
                                       for p in rep[f"{side}_points"]] for side in ("before", "after")}
    out["comparable"] = out["before_same"] and out["after_same"]
    out["after_only"] = out["after_same"] and not out["before_same"]
    if out["comparable"]:
        out["verdict"] = "COMPARABLE"
    else:
        bad = [s for s in ("before", "after") if not out[f"{s}_same"]]
        out["verdict"] = "NOT COMPARABLE (" + " and ".join(f"{s} commits differ" for s in bad) + ")"
    return out


def cross(before_by_col, after_by_col, ref=None, min_ratio=1.5, min_seconds=0.5):
    """One compare() per column, plus the pairing verdict and, only when
    COMPARABLE, each column's ratio against the reference column over the
    lanes both columns compared."""
    names = list(after_by_col)
    missing = [n for n in names if n not in before_by_col] + [n for n in before_by_col if n not in after_by_col]
    if missing:
        raise SystemExit(f"cross: columns need both --before and --after: {sorted(set(missing))}")
    reports = {n: compare(before_by_col[n], after_by_col[n], min_ratio, min_seconds) for n in names}
    pair = pairing(reports)
    ref = ref or names[0]
    if ref not in reports:
        raise SystemExit(f"cross: --ref {ref!r} is not a column ({names})")
    summary = []
    ref_rows = {r["lane"]: r for r in reports[ref]["rows"]}
    for n in names:
        rep = reports[n]
        t = rep["totals"]
        bt, at = sum(t["before"].values()), sum(t["after"].values())
        row = dict(column=n, lanes=len(rep["rows"]), before_s=bt, after_s=at,
                   ratio=(at / bt) if bt > 0 else None, host_factor=rep["host_factor"],
                   regressed=len(rep["regressed"]), improved=len(rep["improved"]),
                   above_host_factor=len(rep["outliers"]),
                   delta=dict((k, t["after"][k] - t["before"][k]) for k in BUCKETS),
                   before=rep["before_points"], after=rep["after_points"],
                   before_cpu=rep["before_host"].get("cpu_model"), after_cpu=rep["after_host"].get("cpu_model"),
                   same_host=rep["same_host"], vs_ref=None)
        if pair["comparable"] and n != ref:
            common = [r for r in rep["rows"] if r["lane"] in ref_rows]
            cb = sum(r["before_s"] for r in common)
            ca = sum(r["after_s"] for r in common)
            rb = sum(ref_rows[r["lane"]]["before_s"] for r in common)
            ra = sum(ref_rows[r["lane"]]["after_s"] for r in common)
            if cb > 0 and rb > 0 and ra > 0:
                row["vs_ref"] = (ca / cb) / (ra / rb)
                row["vs_ref_lanes"] = len(common)
        summary.append(row)
    return dict(pairing=pair, ref=ref, summary=summary, reports=reports)


def series(paths, lanes, stage="non-train", commits=None, common_only=False, columns=None):
    """One lane's seconds per (column, commit), oldest commit first. Each
    point sums the lane's `stage` buckets over the fixtures timed at EVERY
    point of that lane, so a point is never larger for holding more fixtures.
    A point whose lane revision differs from its column's newest point is marked."""
    cells, records = load_side_grouped(paths)
    want = set(columns) if columns else None
    keys = sorted({k for k in records if want is None or k[0] in want},
                  key=lambda k: (_date_of(k[1]) or "9999", k[1], k[0]))
    if commits:
        prefixes = [c for c in commits if c]
        keys = [k for k in keys if any(k[1].startswith(p) for p in prefixes)]
    cols = sorted({k[0] for k in keys})
    commit_order = []
    for _, c in keys:
        if c not in commit_order:
            commit_order.append(c)
    if common_only:
        commit_order = [c for c in commit_order if all((col, c) in records for col in cols)]
        keys = [k for k in keys if k[1] in commit_order]
    buckets = STAGES[stage]
    out = dict(stage=stage, commits=[dict(commit=c[:10], full=c, date=_date_of(c)) for c in commit_order],
               columns=cols, lanes={}, hosts={})
    for k in keys:
        out["hosts"][f"{k[0]}@{k[1][:10]}"] = dict(vendors=sorted(records[k]["vendors"]),
                                                   cpu_models=sorted(records[k]["cpu_models"]))
    for lane in lanes:
        pts = {k: cells.get(k, {}).get(lane) for k in keys}
        pts = {k: v for k, v in pts.items() if v}
        if not pts:
            out["lanes"][lane] = dict(fixtures=[], points={})
            continue
        common = set.intersection(*(set(v["fx"]) for v in pts.values()))
        # Each column's newest point sets that column's lane revision.
        newest = {}
        for k in pts:  # keys are oldest first
            newest[k[0]] = pts[k]["rev"]
        points = {}
        for k, v in pts.items():
            sec = sum(v["fx"][f][b] for f in common for b in buckets)
            points[f"{k[0]}@{k[1][:10]}"] = dict(column=k[0], commit=k[1][:10], seconds=sec,
                                                 revision=v["rev"], revision_differs=v["rev"] != newest[k[0]])
        out["lanes"][lane] = dict(fixtures=sorted(common), points=points)
    return out


def load_side_grouped(paths):
    """Records grouped by (column, commit): cells[(column, commit)][lane] =
    {fx: {fixture: seconds by bucket}, rev}; the last record given supplies a
    cell, as in load_side."""
    cells, records = {}, {}
    for path in _files(paths):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not (isinstance(data, dict) and isinstance(data.get("cells"), dict) and "fixtures" in data):
            continue
        key = (column_of(data.get("vendor")), str(data.get("commit") or ""))
        revs = data.get("lane_revisions") or {}
        timed = 0
        for name, cell in data["cells"].items():
            lane, _, fixture = name.partition("/")
            if not isinstance(cell, dict) or not ran(cell):
                continue
            sec = cell_seconds(cell.get("timing"))
            if sec is None:
                continue
            timed += 1
            e = cells.setdefault(key, {}).setdefault(lane, dict(fx={}, rev=revs.get(lane)))
            e["fx"][fixture] = sec
            e["rev"] = revs.get(lane)
        if timed:
            r = records.setdefault(key, dict(vendors=set(), cpu_models=set()))
            r["vendors"].add(str(data.get("vendor")))
            model = record_host(data).get("cpu_model")
            if model:
                r["cpu_models"].add(model)
    return cells, records


def _fmt(x):
    return "-" if x is None else f"{x:.2f}"


def _point_text(points):
    if not points:
        return "-"
    return "; ".join(f"{p['commit']} ({p['date'] or 'date ?'}, {p['cells']} cells)" for p in points)


def _host_text(host):
    if not host:
        return "-"
    parts = [str(host.get("cpu_model") or "cpu model unrecorded")]
    for key, label in (("logical_cpus", "cpus"), ("cgroup_cpu_quota", "quota"), ("cpu_threads", "threads"),
                       ("concurrency", "shard")):
        if host.get(key) is not None:
            parts.append(f"{label}={host[key]}")
    if not host.get("stated"):
        parts.append("no timing_env")
    if host.get("mixed_cpu_models"):
        parts.append("MIXED " + " / ".join(host["mixed_cpu_models"]))
    return ", ".join(parts)


def _print_text(rep, top):
    print(f"before: {', '.join(rep['before_vendors']) or '-'} at {_point_text(rep['before_points'])}")
    print(f"        host {_host_text(rep['before_host'])}")
    print(f"after:  {', '.join(rep['after_vendors']) or '-'} at {_point_text(rep['after_points'])}")
    print(f"        host {_host_text(rep['after_host'])}")
    if len(rep["before_vendors"]) > 1 or len(rep["after_vendors"]) > 1:
        print("WARNING: one side mixes device labels")
    if len(rep["before_points"]) > 1 or len(rep["after_points"]) > 1:
        print("NOTE: one side mixes commits (reuse or shard records); each lane's commits are in --json")
    if rep["host_differences"]:
        print(("WARNING: hosts differ: " if not rep["same_host"] else "NOTE: host fields unrecorded on one side: ")
              + "; ".join(rep["host_differences"]))
    print(f"host factor (median ratio, lanes >= 0.2 s before): {_fmt(rep['host_factor'])}")
    print(f"{'lane':<40} {'before_s':>9} {'after_s':>9} {'ratio':>7} {'rel':>6} {'delta_s':>8} grew   flag")
    for r in rep["rows"][:top]:
        flag = "REGRESSED" if r["regressed"] else ("faster" if r["improved"] else "")
        if r["outlier"]:
            flag += " outlier"
        if r["batch_rev_changed"]:
            flag += " batch-rev"
        print(f"{r['lane']:<40} {r['before_s']:>9.2f} {r['after_s']:>9.2f} {_fmt(r['ratio']):>7} "
              f"{_fmt(r['rel']):>6} {r['delta_s']:>8.2f} {r['grew']:<6} {flag}")
    for side in ("before", "after"):
        t = rep["totals"][side]
        print(f"TOTAL {side}: " + " ".join(f"{k}={t[k]:.1f}" for k in BUCKETS) + f" all={sum(t.values()):.1f} s")
    print(f"lanes compared {len(rep['rows'])}; regressed {len(rep['regressed'])} ({len(rep['outliers'])} above the host factor too); improved {len(rep['improved'])}; "
          f"skipped (revision/fixture) {len(rep['skipped'])}; only before {len(rep['only_before'])}; "
          f"only after {len(rep['only_after'])}")


def vs_text(res, row):
    """The cross-column cell: `ref`, the ratio when the pairing rule holds
    (marked when the column's own machine changed between its sides), or
    NOT COMPARABLE."""
    if row["column"] == res["ref"]:
        return "ref"
    if not res["pairing"]["comparable"]:
        return "NOT COMPARABLE"
    text = _fmt(row["vs_ref"])
    hosts_moved = [n for n in (row["column"], res["ref"]) if not res["reports"][n]["same_host"]]
    return text + (f" (machine changed: {', '.join(hosts_moved)})" if hosts_moved else "")


def _cross_host(res, row):
    rep = res["reports"][row["column"]]
    b = ",".join(rep["before_vendors"]) + (f" [{row['before_cpu']}]" if row["before_cpu"] else "")
    a = ",".join(rep["after_vendors"]) + (f" [{row['after_cpu']}]" if row["after_cpu"] else "")
    return b if a == b else f"{b} -> {a}"


def _print_cross(res, top):
    pair = res["pairing"]
    print("| column | before commit (date) | after commit (date) | host before -> after | lanes | before s "
          "| after s | ratio | host factor | regressed | above host factor | faster | d train | d infer | d other "
          "| vs " + res["ref"] + " |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in res["summary"]:
        print(f"| {r['column']} | {_point_text(r['before'])} | {_point_text(r['after'])} | {_cross_host(res, r)} "
              f"| {r['lanes']} | {r['before_s']:.0f} | {r['after_s']:.0f} | {_fmt(r['ratio'])} "
              f"| {_fmt(r['host_factor'])} | {r['regressed']} | {r['above_host_factor']} | {r['improved']} "
              f"| {r['delta']['train']:+.0f} | {r['delta']['infer']:+.0f} | {r['delta']['other']:+.0f} "
              f"| {vs_text(res, r)} |")
    print()
    print(f"CROSS-COLUMN: {pair['verdict']}")
    for side in ("before", "after"):
        state = "same commit on every column" if pair[f"{side}_same"] else "commits differ"
        print(f"  {side}: {state}: " + "; ".join(
            f"{n} {_point_text(c[side])}" for n, c in pair["columns"].items()))
    if pair["after_only"]:
        print("  The after side alone is comparable across columns; a statement about the change since "
              "'before' is not.")
    if top:
        for n, rep in res["reports"].items():
            print(f"\n## {n}")
            _print_text(rep, top)


def _print_series(res):
    print(f"stage: {res['stage']} (seconds summed over the fixtures timed at every point of the lane)")
    head = " | ".join(f"{c['date'] or 'date ?'} {c['commit']}" for c in res["commits"])
    print(f"| lane | column | {head} |")
    print("|---|---|" + "---:|" * len(res["commits"]))
    for lane, e in res["lanes"].items():
        first = True
        for col in res["columns"]:
            vals = []
            for c in res["commits"]:
                p = e["points"].get(f"{col}@{c['commit']}")
                vals.append("-" if p is None else (f"{p['seconds']:.2f}" + ("*" if p["revision_differs"] else "")))
            if all(v == "-" for v in vals):
                continue
            print(f"| {lane if first else ''} | {col} | " + " | ".join(vals) + " |")
            first = False
        if first:
            print(f"| {lane} | - | " + " | ".join("-" for _ in res["commits"]) + " |")
    print("\n* = lane revision differs from the column's newest point (fixture or arithmetic moved).")
    print("hosts: " + "; ".join(f"{k} {','.join(v['vendors'])}" + (f" [{','.join(v['cpu_models'])}]" if v["cpu_models"] else "")
                                for k, v in res["hosts"].items()))


def _col_paths(items, flag):
    out = {}
    for item in items or []:
        name, sep, paths = item.partition("=")
        if not sep or not name or not paths:
            raise SystemExit(f"cross: {flag} takes COL=<path>[,<path>...], got {item!r}")
        out.setdefault(name, []).extend(p for p in paths.split(",") if p)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "cross":
        p = argparse.ArgumentParser(prog="identity_timing_compare.py cross")
        p.add_argument("--before", action="append", required=True, help="COL=<record|dir>[,...]; repeat per column")
        p.add_argument("--after", action="append", required=True, help="COL=<record|dir>[,...]; repeat per column")
        p.add_argument("--ref", default=None, help="the column the others are set against (default: first)")
        p.add_argument("--min-ratio", type=float, default=1.5)
        p.add_argument("--min-seconds", type=float, default=0.5)
        p.add_argument("--top", type=int, default=0, help="also print each column's top N lanes")
        p.add_argument("--commit-dates", default=None, help="<commit>\\t<date> file to seed dates")
        p.add_argument("--json", action="store_true")
        args = p.parse_args(argv[1:])
        if args.commit_dates:
            load_commit_dates(args.commit_dates)
        res = cross(_col_paths(args.before, "--before"), _col_paths(args.after, "--after"), args.ref,
                    args.min_ratio, args.min_seconds)
        if args.json:
            json.dump(res, sys.stdout, indent=1, default=sorted)
            print()
        else:
            _print_cross(res, args.top)
        ok = all(r["before_records"] and r["after_records"] for r in res["reports"].values())
        return 0 if ok else 1
    if argv and argv[0] == "series":
        p = argparse.ArgumentParser(prog="identity_timing_compare.py series")
        p.add_argument("paths", nargs="+", help="records or dirs (any columns, any commits)")
        p.add_argument("--lane", action="append", required=True)
        p.add_argument("--stage", choices=sorted(STAGES), default="non-train")
        p.add_argument("--commits", default="", help="comma-separated commit prefixes to keep")
        p.add_argument("--columns", default="", help="comma-separated columns to keep (nvidia,amd,apple,cpu,ptx)")
        p.add_argument("--common-only", action="store_true", help="only commits recorded on every kept column")
        p.add_argument("--commit-dates", default=None, help="<commit>\\t<date> file to seed dates")
        p.add_argument("--json", action="store_true")
        args = p.parse_args(argv[1:])
        if args.commit_dates:
            load_commit_dates(args.commit_dates)
        res = series(args.paths, args.lane, args.stage, [c for c in args.commits.split(",") if c],
                     args.common_only, [c for c in args.columns.split(",") if c] or None)
        if args.json:
            json.dump(res, sys.stdout, indent=1)
            print()
        else:
            _print_series(res)
        return 0 if res["commits"] else 1
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", nargs="+", required=True, help="records or dirs, oldest first")
    parser.add_argument("--after", nargs="+", required=True, help="records or dirs, oldest first")
    parser.add_argument("--min-ratio", type=float, default=1.5)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--commit-dates", default=None, help="<commit>\\t<date> file to seed dates")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.commit_dates:
        load_commit_dates(args.commit_dates)
    rep = compare(args.before, args.after, args.min_ratio, args.min_seconds)
    if args.json:
        json.dump(rep, sys.stdout, indent=1)
        print()
    else:
        _print_text(rep, args.top)
    return 0 if rep["before_records"] and rep["after_records"] else 1


if __name__ == "__main__":
    sys.exit(main())
