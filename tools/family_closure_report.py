#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HOW MUCH OF THE SOURCE IS REALLY SHARED BETWEEN RELEASE FAMILIES (AGENTS.md part 4, Andrew 2026-11-28).

    python3 tools/family_closure_report.py [--commit HEAD] [--json OUT] [--top 80] [--cache-dir DIR]

A release rebuilds a family when a file in its source closure changes (tools/release_reuse.py identity;
tools/release_families.py makes the rows whole per family). A file in the closure of several families rebuilds all
of them, so this report says, per family, how many files its closure holds and how many of them it shares, and then
lists every file in the closure of MORE than one family, sorted by how many families it reaches (then by path),
with a per-directory roll-up. It answers whether a one-family change rebuilds one family or all of them, and which
files to move out of the shared core ("code used by one family lives in that family").

THE CLOSURES are tools/release_reuse.py's: `git archive` of the commit (never a working tree), one
`bincache.source_digest` walk per build script, run across every core of this machine by release_reuse.tree_facts
(a process per script; MOJOLEARN_REUSE_WORKERS caps it; a walk holds well under 200 MB), and cached by commit under
--cache-dir, the same cache the release plan uses. A family's closure is the union of its bindings' closures (Linux
and macOS share the build scripts). The merged-file ties between families (bindings/merged_groups.json) are not
source and are not counted here; tools/release_families.py apply reports them as "shares the merged file".

It builds, rents and records nothing.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def family_closures(facts, fam, names):
    """({family: set(files)}, {family: [binding]}, [binding with no closure]) from release_reuse tree facts."""
    import release_reuse as RR
    closures, members, missing = {}, {}, []
    for name in sorted(names):
        f = fam.binding(name)
        if f is None:
            continue
        script = RR.Binding(RR.LINUX, "host", "", "host", name).script
        c = (facts.get("closures") or {}).get(script)
        members.setdefault(f, []).append(name)
        if c is None:
            missing.append(name)
            continue
        closures.setdefault(f, set()).update(c.get("sources") or [])
    for f in fam.names():
        closures.setdefault(f, set())
        members.setdefault(f, [])
    return closures, members, missing


def report(closures):
    """The report document: per family totals, every file shared by more than one family, the directory roll-up."""
    reach = {}
    for f, files in closures.items():
        for path in files:
            reach.setdefault(path, set()).add(f)
    shared = sorted(((p, sorted(fs)) for p, fs in reach.items() if len(fs) > 1), key=lambda kv: (-len(kv[1]), kv[0]))
    per_family = {}
    for f, files in closures.items():
        mine = sum(1 for p in files if len(reach[p]) == 1)
        per_family[f] = dict(files=len(files), own=mine, shared=len(files) - mine,
                             shared_with_all=sum(1 for p in files if len(reach[p]) == len(closures)))
    dirs = {}
    for path, fs in shared:
        d = path.split("/", 1)[0] if "/" in path else "."
        row = dirs.setdefault(d, dict(files=0, max_families=0))
        row["files"] += 1
        row["max_families"] = max(row["max_families"], len(fs))
    return dict(families=per_family, shared=[dict(path=p, families=fs, reach=len(fs)) for p, fs in shared],
                dirs=dict(sorted(dirs.items(), key=lambda kv: -kv[1]["files"])), total_files=len(reach))


def lines(doc, commit, top, members, missing):
    out = ["== family closures at %s: %d files in all, %d in the closure of more than one family" % (
        commit[:12], doc["total_files"], len(doc["shared"]))]
    out.append("   %-10s %6s %6s %6s %8s  %s" % ("family", "files", "own", "shared", "in all", "bindings"))
    for f, row in sorted(doc["families"].items(), key=lambda kv: -kv[1]["files"]):
        out.append("   %-10s %6d %6d %6d %8d  %d" % (f, row["files"], row["own"], row["shared"],
                                                      row["shared_with_all"], len(members.get(f) or [])))
    out.append("== shared files by directory (files, most families reached)")
    for d, row in doc["dirs"].items():
        out.append("   %-24s %6d  %d" % (d, row["files"], row["max_families"]))
    out.append("== files in more than one family's closure, most families first (top %d of %d)" % (
        min(top, len(doc["shared"])), len(doc["shared"])))
    for row in doc["shared"][:top]:
        out.append("   %2d  %-70s %s" % (row["reach"], row["path"], ",".join(row["families"])))
    if missing:
        out.append("note: %d binding(s) have no build script at this commit: %s" % (len(missing), ", ".join(missing[:12])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--json", default="", help="write the whole report (every shared file) here")
    ap.add_argument("--top", type=int, default=80, help="shared files printed (default 80; --json has all)")
    ap.add_argument("--cache-dir", default=os.environ.get("MOJOLEARN_REUSE_CACHE")
                    or str(Path.home() / "mojolearn-evidence" / "release" / "identities"),
                    help="release_reuse's facts cache (the release plan's own, by default)")
    a = ap.parse_args(argv)
    import release_families as RF
    import release_reuse as RR
    commit = RR.git("rev-parse", a.commit + "^{commit}").strip()
    fam = RF.load()
    facts = RR.facts_for_commit(commit, a.cache_dir)   # every build script's closure, across every core, cached
    names = RF.known_bindings()
    closures, members, missing = family_closures(facts, fam, names)
    doc = report(closures)
    doc.update(commit=commit, members=members, missing=missing)
    for line in lines(doc, commit, a.top, members, missing):
        print(line)
    if a.json:
        RR.atomic_write_text(a.json, json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print("== json: " + a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
