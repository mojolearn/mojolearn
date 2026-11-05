#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE LAYOUT go/no-go: ship the development group layout, or keep the merged one.

    python3 tools/one_layout.py measure --release-dir ~/mojolearn-evidence/release/0.8.65/6519a54c6579 \
        [--dev-cache ~/.cache/mojolearn-development/artifacts] [--pypi-usage usage.json] \
        [--new HEAD] [--since 2026-10-04] [--no-closures] --json one-layout.json
    python3 tools/one_layout.py verdict --json one-layout.json

Static, from retained evidence only; it never compiles, installs or uploads:

- the retained release wheels (`<release-dir>/macos/*.whl`, `<release-dir>/linux/*.whl`): file size,
  native files and their compressed bytes per set directory (`identical/`, `host/`);
- the retained merged artifacts (`<release-dir>/macos-bincache/*.tar.gz`) and the development
  cache (`tools/development_build.py`, one `receipt.json` per artifact), deflated the same way
  (zlib level 6) so the two layouts compare like for like; the measured ratio per set is what
  turns each merged wheel into a development-layout estimate;
- the import closures of both layouts (`tools/compile_closures.py`, `tools/merged_bindings.py
  plan --layout release|development`): groups, full-build files, commit-weighted closure files
  compiled per changed file;
- the per-project PyPI limits (packaging/pypi_limits.json) and, when given, each project's bytes
  already on PyPI (`--pypi-usage`, {distribution: bytes}).

A development set whose artifacts are not all in the cache is estimated two ways: `central`
fills each missing member with the median measured member of that set, `pessimistic` grows each
merged group by the worst development/merged ratio measured on any fully measured group. The
verdict uses `pessimistic` for GO; a wheel that fits only under `central` is HOLD (measure it).
"""
import argparse
import collections
import datetime
import json
import os
import statistics
import sys
import tarfile
import zipfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "packaging"))
import pypi_limits  # noqa: E402

SCHEMA = "mojolearn.one-layout.v1"
SETS = ("identical", "host")
NATIVE = (".so", ".dylib")


def set_of(member):
    """The set directory a wheel member belongs to: identical, host or None (runtime, adapters)."""
    parts = member.split("/")
    for s in SETS:
        if s in parts[:-1]:
            return s
    return None


def wheel_rows(paths):
    """[{wheel, distribution, file_bytes, sets: {set: {files, compressed, uncompressed}}}]."""
    rows = []
    for p in sorted(paths):
        p = Path(p)
        sets = collections.defaultdict(lambda: dict(files=0, compressed=0, uncompressed=0))
        with zipfile.ZipFile(p) as z:
            for i in z.infolist():
                if i.filename.endswith(NATIVE) and set_of(i.filename):
                    row = sets[set_of(i.filename)]
                    row["files"] += 1
                    row["compressed"] += i.compress_size
                    row["uncompressed"] += i.file_size
                    stem = i.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                    row.setdefault("stems", {})[stem] = i.compress_size
        rows.append(dict(wheel=p.name, distribution=pypi_limits.wheel_distribution(p),
                         platform="darwin" if "macosx" in p.name else "linux",
                         file_bytes=p.stat().st_size, sets=dict(sets)))
    return rows


def deflated(data):
    return len(zlib.compress(data, 6))


def merged_artifacts(bincache_dir):
    """{stem: {bytes, deflate}} of every merged .so in a retained bincache (tar.gz archives)."""
    out = {}
    for t in sorted(Path(bincache_dir).glob("*.tar.gz")):
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(NATIVE):
                    data = tf.extractfile(m).read()
                    stem = Path(m.name).name.rsplit(".", 1)[0]
                    out[stem] = dict(bytes=len(data), deflate=deflated(data))
    return out


def dev_artifacts(cache_dir):
    """{stem: {mode, bytes, deflate, peak_rss_bytes, seconds}}: the newest artifact per stem."""
    best = {}
    for receipt in Path(cache_dir).rglob("receipt.json"):
        try:
            rec = json.loads(receipt.read_text())
        except (OSError, ValueError):
            continue
        if "stem" not in rec or "mode" not in rec:
            continue  # support artifacts (libMojolearnMath) are the same in both layouts
        libs = [p for p in receipt.parent.iterdir() if p.name.endswith(NATIVE)]
        if not libs:
            continue
        stamp = receipt.stat().st_mtime
        if rec["stem"] in best and best[rec["stem"]][0] >= stamp:
            continue
        data = libs[0].read_bytes()
        best[rec["stem"]] = (stamp, dict(mode=rec["mode"], bytes=len(data), deflate=deflated(data),
                                         peak_rss_bytes=rec.get("peak_rss_bytes"),
                                         seconds=(rec.get("execution") or {}).get("elapsed_seconds")))
    return {k: v for k, (_, v) in best.items()}


def layout_plans(tree, plat):
    """{layout: {set: [(stem, members)]}} from tools/merged_bindings.py."""
    import merged_bindings
    out = {}
    for layout in ("release", "development"):
        out[layout] = {s: [(stem, list(mem)) for stem, _cls, mem in
                           merged_bindings.plan(s, plat, True, layout, repo=tree)] for s in SETS}
    return out


def set_ratio(release_groups, dev_groups, merged, dev, worst_seen=1.0):
    """Development/merged size ratio of one set (deflated bytes).

    `worst_seen` is the worst ratio of a fully measured group in another set: a set with no
    fully measured group (the host set on 0.8.65's Mac cache) is grown by it.
    Returns dict(measured, missing, merged, dev_measured, central, pessimistic, worst_group_ratio)."""
    owner = {m: stem for stem, mem in dev_groups for m in mem}
    merged_total = sum(merged.get(stem, {}).get("deflate", 0) for stem, _ in release_groups)
    measured = [stem for stem, _ in dev_groups if stem in dev]
    missing = [stem for stem, _ in dev_groups if stem not in dev]
    dev_measured = sum(dev[s]["deflate"] for s in measured)
    median = statistics.median([dev[s]["deflate"] for s in measured]) if measured else 0
    worst = 1.0
    per_group = []
    for stem, mem in release_groups:
        ds = sorted({owner[m] for m in mem if m in owner})
        if stem in merged and ds and all(d in dev for d in ds):
            r = sum(dev[d]["deflate"] for d in ds) / max(1, merged[stem]["deflate"])
            per_group.append((stem, round(r, 3)))
            worst = max(worst, r)
    own_worst = worst
    worst = max(worst, worst_seen)
    central = dev_measured + median * len(missing)
    pessimistic = 0
    for stem, mem in release_groups:
        ds = sorted({owner[m] for m in mem if m in owner})
        known = sum(dev[d]["deflate"] for d in ds if d in dev)
        if all(d in dev for d in ds):
            pessimistic += known
        else:
            pessimistic += max(known, worst * merged.get(stem, {}).get("deflate", 0))
    den = max(1, merged_total)
    return dict(dev_groups=len(dev_groups), release_groups=len(release_groups), measured=len(measured),
                missing=missing, merged_deflate=merged_total, dev_measured_deflate=dev_measured,
                central_ratio=round(central / den, 3), pessimistic_ratio=round(pessimistic / den, 3),
                worst_group_ratio=round(own_worst, 3), grown_by=round(worst, 3), per_group=per_group)


def estimate_wheels(wheels, ratios):
    """Development-layout size estimates of each retained wheel: every set's compressed native
    bytes scaled by that set's ratio; everything else (Python, runtime, metadata) unchanged."""
    out = []
    for w in wheels:
        rest = w["file_bytes"] - sum(s["compressed"] for s in w["sets"].values())
        est = {}
        for kind in ("central", "pessimistic"):
            total = rest
            for s, row in w["sets"].items():
                r = ratios.get(s, {}).get(kind + "_ratio", 1.0)
                if kind == "pessimistic" and w["platform"] != "darwin":
                    # the Linux groups differ from the Mac's: grow them by the worst measured group
                    r = max(r, ratios.get(s, {}).get("grown_by", 1.0))
                total += row["compressed"] * r
            est[kind] = int(total)
        files = {s: row["files"] for s, row in w["sets"].items()}
        out.append(dict(w, estimate=est, native_files_release=sum(files.values())))
    return out


def load_cost(plans):
    """dlopens for a full import (every member loaded) and loader index entries, per layout."""
    out = {}
    for layout, sets in plans.items():
        out[layout] = dict(native_files=sum(len(g) for g in sets.values()),
                           members=sum(len(m) for g in sets.values() for _, m in g),
                           index_bytes=sum(len(json.dumps({"files": {stem + ".so": sorted(m) for stem, m in g}}))
                                           for g in sets.values()))
    return out


def closure_scope(tree, plat, since=None, repo=ROOT):
    """{layout: {set: dict(groups, full_files, max_files, expected_files_per_change)}}."""
    import compile_closures as cc
    plans = layout_plans(tree, plat)
    every = sorted({m for lay in plans.values() for g in lay.values() for _, mem in g for m in mem})
    mc = cc.member_closures(tree, every)
    out = {}
    weights = None
    for layout, sets in plans.items():
        out[layout] = {}
        for s, groups in sets.items():
            grouping = {stem: {r for m in mem for r in mc[m]} for stem, mem in groups}
            if weights is None and since:
                files = {r for c in mc.values() for r in c}
                weights, _ = cc.change_weights(repo, since, files)
            wd = {f: w for f, w in (weights or {}).items() if any(f in g for g in grouping.values())}
            out[layout][s] = dict(groups=len(grouping), full_files=sum(len(g) for g in grouping.values()),
                                  max_files=max(len(g) for g in grouping.values()),
                                  expected_files_per_change=round(cc.expected_scope(grouping, wd), 1) if wd else None)
    return out


def verdict(doc, cfg=None):
    """GO / HOLD / NO-GO with the reasons, from a measure document."""
    cfg = cfg or pypi_limits.load()
    usage = doc.get("pypi_usage") or {}
    reasons, rows = [], []
    status = "GO"
    per_release = collections.defaultdict(lambda: dict(central=0, pessimistic=0))
    for w in doc["wheels"]:
        limit = pypi_limits.file_limit(w["distribution"], cfg)
        c, p = w["estimate"]["central"], w["estimate"]["pessimistic"]
        per_release[w["distribution"]]["central"] += c
        per_release[w["distribution"]]["pessimistic"] += p
        state = "fits" if p <= limit else ("fits only the central estimate" if c <= limit else "over")
        rows.append(dict(wheel=w["wheel"], distribution=w["distribution"], release_bytes=w["file_bytes"],
                         central_bytes=c, pessimistic_bytes=p, limit_bytes=limit, state=state,
                         margin_central=round(1 - c / limit, 3)))
        if state == "over":
            status = "NO-GO"
            reasons.append(f"{w['wheel']}: development layout {c / 2**20:.1f} MiB > {limit / 2**20:.0f} MiB")
        elif state != "fits" and status == "GO":
            status = "HOLD"
            reasons.append(f"{w['wheel']}: {c / 2**20:.1f} MiB (central) fits {limit / 2**20:.0f} MiB, "
                           f"{p / 2**20:.1f} MiB (pessimistic) does not: measure the missing development "
                           "artifacts of this platform before switching")
    projects = []
    for dist, est in sorted(per_release.items()):
        used = usage.get(dist)
        plimit = pypi_limits.project_limit(dist, cfg)
        if used is None:
            continue
        left = plimit - used
        projects.append(dict(distribution=dist, used_bytes=used, limit_bytes=plimit,
                             releases_left_release_layout=None,
                             releases_left_central=int(left // max(1, est["central"]))))
        if used + est["central"] > plimit:
            status = "NO-GO"
            reasons.append(f"{dist}: project size {used / 2**30:.2f} GiB + one release exceeds "
                           f"{plimit / 2**30:.0f} GiB")
    for p in projects:
        rel = sum(w["file_bytes"] for w in doc["wheels"] if w["distribution"] == p["distribution"])
        p["releases_left_release_layout"] = int((p["limit_bytes"] - p["used_bytes"]) // max(1, rel))
    return dict(status=status, reasons=reasons, wheels=rows, projects=projects,
                bits="unknown: the compiler documents no guarantee that a function compiles to the same "
                     "machine code in a different `mojo build` unit; switch, then re-record every lane on every "
                     "vendor together once (allowed within a version)")


def measure(args):
    rel = Path(args.release_dir).expanduser()
    wheels = wheel_rows(list(rel.glob("macos/*.whl")) + list(rel.glob("linux/*.whl")))
    merged = merged_artifacts(rel / "macos-bincache") if (rel / "macos-bincache").is_dir() else {}
    dev = dev_artifacts(Path(args.dev_cache).expanduser()) if args.dev_cache else {}
    import bincache
    tree = bincache.revision_tree(ROOT, args.new)
    doc = dict(schema=SCHEMA, release_dir=str(rel), new=args.new, measured_at=datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds"), platforms={})
    ratios = {}
    for plat in ("darwin", "linux"):
        plans = layout_plans(tree, plat)
        entry = dict(load=load_cost(plans))
        if plat == "darwin":
            first = {s: set_ratio(plans["release"][s], plans["development"][s], merged, dev) for s in SETS}
            worst = max(r["worst_group_ratio"] for r in first.values())
            for s in SETS:
                ratios[s] = set_ratio(plans["release"][s], plans["development"][s], merged, dev, worst)
            entry["ratios"] = ratios
        if not args.no_closures:
            entry["closures"] = closure_scope(tree, plat, args.since)
        doc["platforms"][plat] = entry
    # Linux has no retained development artifacts: the darwin ratio per set is its estimate.
    doc["wheels"] = estimate_wheels(wheels, ratios)
    if args.pypi_usage:
        doc["pypi_usage"] = {pypi_limits.normalize(k): int(v)
                             for k, v in json.loads(Path(args.pypi_usage).read_text()).items()}
    doc["verdict"] = verdict(doc)
    Path(args.json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print_verdict(doc)
    return 0


def print_verdict(doc):
    v = doc["verdict"]
    print(f"ONE LAYOUT: {v['status']}")
    for r in v["wheels"]:
        print(f"  {r['state']:<32} {r['wheel']}: release {r['release_bytes'] / 2**20:.1f} MiB -> development "
              f"{r['central_bytes'] / 2**20:.1f} (central) / {r['pessimistic_bytes'] / 2**20:.1f} (pessimistic) MiB, "
              f"limit {r['limit_bytes'] / 2**20:.0f} MiB")
    for p in v["projects"]:
        print(f"  project {p['distribution']}: {p['used_bytes'] / 2**30:.2f} of {p['limit_bytes'] / 2**30:.0f} GiB; "
              f"releases left {p['releases_left_release_layout']} (release layout) -> {p['releases_left_central']}")
    for r in v["reasons"]:
        print("  reason: " + r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--release-dir", required=True)
    m.add_argument("--dev-cache", default=str(Path.home() / ".cache/mojolearn-development/artifacts"))
    m.add_argument("--pypi-usage", help="JSON {distribution: bytes already on PyPI}")
    m.add_argument("--new", default="HEAD")
    m.add_argument("--since", default=None, help="weight changed files by commits since this date")
    m.add_argument("--no-closures", action="store_true")
    m.add_argument("--json", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "measure":
        return measure(args)
    doc = json.loads(Path(args.json).read_text())
    doc["verdict"] = verdict(doc)
    print_verdict(doc)
    return 0 if doc["verdict"]["status"] == "GO" else 3


if __name__ == "__main__":
    sys.exit(main())
