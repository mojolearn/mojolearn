#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Static compile-closure report: what each compile reads, what a change rebuilds.

    python3 tools/compile_closures.py report [--old REV] [--new REV] [--since DATE]
        [--release-keys OLD_DIR NEW_DIR] [--json FILE] [--out docs/build/COMPILE_CLOSURES.md]

Build tooling; never compiles, never runs a box. Everything comes from
`git archive` of committed revisions (tools/bincache.py revision_tree), the
closure-v2 walk that keys the binary cache (tools/bincache.py import_closure,
the same `-I . -I bindings` resolution `mojo build` uses), the merged plan
(tools/merged_bindings.py) and, optionally, the bincache keys/<key>.json two
releases kept, to explain why their keys differed.

The compile-scope proxy is the number of distinct Mojo source files a compile
reads (its closure). It is not a time: compile time also depends on how much of
each file is instantiated. Where the releases kept measured seconds the report
prints them beside the closure size.
"""
import argparse
import collections
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bincache  # noqa: E402
import merged_bindings  # noqa: E402

ROOT = HERE.parent
INCS = (".", "bindings")
PLATFORMS = ("darwin", "linux")
SETS = ("identical", "host")
#: The per-binding scripts that are not one binding's compile.
NOT_BINDING_SCRIPTS = ("build_defines.sh", "build_host_family.sh", "merged_build.sh")


def closure_of(tree, roots):
    """{rel: sha} of the Mojo closure of ROOTS (generated sources: their generator)."""
    files, generated = bincache.import_closure(tree, roots, [tree / i for i in INCS])
    files.update(bincache.generator_inputs(tree, generated))
    return files


def member_closures(tree, members):
    return {m: closure_of(tree, [tree / "bindings" / (m + ".mojo")]) for m in members}


def group_rows(tree, plat, layout="release"):
    """[(set, stem, flags_class, members)] for one platform's release build."""
    rows = []
    for d in SETS:
        for stem, cls, mem in merged_bindings.plan(d, plat, True, layout, repo=tree):
            rows.append((d, stem, cls, list(mem)))
    return rows


def groups(tree, plat, mc):
    """{(set, stem): dict(flags, members, closure)} with member closures MC."""
    out = {}
    for d, stem, cls, mem in group_rows(tree, plat):
        files = {}
        for m in mem:
            files.update(mc[m])
        out[(d, stem)] = dict(flags=cls, members=mem, closure=files)
    return out


def script_closures(tree):
    """{script: dict(files, unresolved, note)} for every per-binding build script."""
    out = {}
    for s in sorted((tree / "bindings").glob("build*.sh")):
        if s.name in NOT_BINDING_SCRIPTS:
            continue
        plan = bincache.script_plan(tree, s, [])
        if plan is None:
            out[s.name] = dict(files=None, unresolved=None, note="compile line unreadable")
            continue
        roots, incs, _ = plan
        try:
            files, generated = bincache.import_closure(tree, roots, incs)
            out[s.name] = dict(files=len(files) + len(bincache.generator_inputs(tree, generated)),
                               unresolved=None, note="")
        except bincache.UnresolvedImport as exc:
            out[s.name] = dict(files=None, unresolved="%s in %s" % (exc.module, exc.importer), note="")
    return out


def v1_merged_scope(tree):
    """What the v1 key hashed for a merged build: script_plan reads no root
    from merged_build.sh's compile line, so the whole tree."""
    plan = bincache.script_plan(tree, tree / "bindings" / "merged_build.sh", ["gpu"])
    return dict(readable=plan is not None, tree_files=len(bincache.tree_sources(tree)))


def shared_ranking(gs):
    reach = collections.Counter()
    for g in gs.values():
        for rel in g["closure"]:
            if rel.endswith(".mojo"):
                reach[rel] += 1
    return reach


def change_weights(repo, since, files):
    """{rel: commits touching it since SINCE} over FILES (git log, first parent off)."""
    out = subprocess.run(["git", "-C", str(repo), "log", "--since=" + since, "--name-only", "--format=",
                          "--", "*.mojo"], capture_output=True, text=True, check=True).stdout
    c = collections.Counter(line.strip() for line in out.splitlines() if line.strip())
    n = subprocess.run(["git", "-C", str(repo), "rev-list", "--count", "--since=" + since, "HEAD", "--", "*.mojo"],
                       capture_output=True, text=True, check=True).stdout.strip()
    return {f: c.get(f, 0) for f in files}, int(n or 0)


def scope(grouping, changed):
    """(groups compiled, closure files compiled) when CHANGED files change."""
    hit = [g for g, files in grouping.items() if changed & files]
    return len(hit), sum(len(grouping[g]) for g in hit)


def expected_scope(grouping, weights):
    """Commit-weighted mean closure files compiled per changed-file event."""
    total = sum(weights.values()) or 1
    cost = 0
    for f, w in weights.items():
        if w:
            cost += w * sum(len(files) for files in grouping.values() if f in files)
    return cost / total


def regroup(mc, current, max_closure=None):
    """Cluster members by closure overlap: start from one group per member and
    merge, each step, the pair that saves the most shared compile work per unit
    of added incremental scope (sum of squared closure sizes), until the full
    build reads no more files than CURRENT does. A group never exceeds
    MAX_CLOSURE files (default: the largest current group), the memory proxy."""
    groups_ = {(m,): set(c) for m, c in mc.items()}
    budget = sum(len(f) for f in current.values())
    cap = max_closure or max(len(f) for f in current.values())
    while len(groups_) > 1:
        over = sum(len(f) for f in groups_.values()) > budget
        best = None
        keys = list(groups_)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                union = groups_[a] | groups_[b]
                if len(union) > cap:
                    continue
                saved = len(groups_[a]) + len(groups_[b]) - len(union)
                grown = len(union) ** 2 - len(groups_[a]) ** 2 - len(groups_[b]) ** 2
                score = saved / max(grown, 1)
                if best is None or score > best[0]:
                    best = (score, a, b, union)
        if best is None:
            break
        # Under the budget, keep merging only pairs that add no incremental
        # scope (one closure inside the other): pure savings.
        if not over:
            zero = [(a, b, groups_[a] | groups_[b]) for i, a in enumerate(keys) for b in keys[i + 1:]
                    if len(groups_[a] | groups_[b]) == max(len(groups_[a]), len(groups_[b]))]
            if not zero:
                break
            best = (0,) + zero[0]
        _, a, b, union = best
        del groups_[a], groups_[b]
        groups_[tuple(sorted(a + b))] = union
    return groups_


def diff_closures(old, new):
    return sorted(r for r in set(old) | set(new) if old.get(r) != new.get(r))


def key_reasons(old_dir, new_dir):
    """{args: [changed field names]} between two releases' bincache keys/<key>.json."""
    def load(d):
        out = {}
        for p in sorted(Path(d).glob("*.json")):
            f = json.loads(p.read_text())
            out.setdefault(" ".join(f.get("args", [])), f)   # first build of each group in the run
        return out
    a, b = load(old_dir), load(new_dir)
    out = {}
    for args in sorted(set(a) & set(b)):
        out[args] = dict(changed=bincache._changed(a[args], b[args]),
                         old_key=bincache.key_of(a[args])[:16], new_key=bincache.key_of(b[args])[:16],
                         old_scope=a[args]["source"].get("scope"), new_scope=b[args]["source"].get("scope"),
                         old_files=a[args]["source"].get("files"), new_files=b[args]["source"].get("files"))
    return out


def analyse(repo, old, new, since, release_keys=None):
    repo = Path(repo).resolve()
    t_new = bincache.revision_tree(repo, new)
    t_old = bincache.revision_tree(repo, old) if old else None
    every = sorted({m for plat in PLATFORMS for _, _, _, mem in group_rows(t_new, plat) for m in mem})
    mc = member_closures(t_new, every)
    report = dict(new=new, old=old, since=since, platforms={})
    old_mc = None
    if t_old is not None:
        old_members = sorted({m for plat in PLATFORMS for _, _, _, mem in group_rows(t_old, plat) for m in mem})
        old_mc = member_closures(t_old, old_members)
    all_files = set()
    for plat in PLATFORMS:
        gs = groups(t_new, plat, mc)
        all_files |= {r for g in gs.values() for r in g["closure"]}
        rows = []
        old_gs = groups(t_old, plat, old_mc) if old_mc is not None else {}
        for (d, stem), g in gs.items():
            row = dict(set=d, stem=stem, flags=g["flags"], members=g["members"],
                       files=sum(1 for r in g["closure"] if r.endswith(".mojo")))
            if (d, stem) in old_gs:
                row["changed_since_old"] = [r for r in diff_closures(old_gs[(d, stem)]["closure"], g["closure"])]
            rows.append(row)
        reach = shared_ranking(gs)
        n = len(gs)
        report["platforms"][plat] = dict(groups=rows, n_groups=n,
                                         shared_by_all=sorted(r for r, c in reach.items() if c == n),
                                         ranking=reach.most_common(60))
    weights, n_commits = change_weights(repo, since, all_files)
    report["weights"] = dict(commits_with_mojo=n_commits,
                             top=sorted(((w, f) for f, w in weights.items() if w), reverse=True)[:40])
    # The files that cost the most rebuilds: commits since SINCE x groups reached (Linux, 9 groups).
    linux_reach = dict(report["platforms"]["linux"]["ranking"])
    full = shared_ranking(groups(t_new, "linux", mc))
    report["hot"] = sorted(((weights.get(f, 0) * c, weights.get(f, 0), c, f) for f, c in full.items()
                            if weights.get(f, 0)), reverse=True)[:25]
    del linux_reach
    report["scripts"] = script_closures(t_new)
    report["v1_merged"] = v1_merged_scope(t_new)
    report["member_files"] = {m: sum(1 for r in c if r.endswith(".mojo")) for m, c in mc.items()}
    # regrouping, per platform and set
    report["regroup"] = {}
    for plat in PLATFORMS:
        for d in SETS:
            cur = {stem: set(g["closure"]) for (dd, stem), g in groups(t_new, plat, mc).items() if dd == d}
            mem = sorted({m for (dd, _), g in groups(t_new, plat, mc).items() if dd == d for m in g["members"]})
            # One group's members on macOS split by flags class; cluster within a class.
            classes = collections.defaultdict(list)
            for (dd, stem), g in groups(t_new, plat, mc).items():
                if dd == d:
                    classes[g["flags"]].extend(g["members"])
            proposal = {}
            for cls, names in classes.items():
                cur_cls = {stem: set(g["closure"]) for (dd, stem), g in groups(t_new, plat, mc).items()
                           if dd == d and g["flags"] == cls}
                for k, v in regroup({m: mc[m] for m in names}, cur_cls).items():
                    proposal[(cls,) + k] = v
            wd = {f: w for f, w in weights.items() if any(f in s for s in cur.values())}
            report["regroup"]["%s/%s" % (plat, d)] = dict(
                members=len(mem),
                current=dict(groups=len(cur), full_files=sum(len(s) for s in cur.values()),
                             max_files=max(len(s) for s in cur.values()),
                             expected=round(expected_scope(cur, wd), 1)),
                proposed=dict(groups=len(proposal), full_files=sum(len(s) for s in proposal.values()),
                              max_files=max(len(s) for s in proposal.values()),
                              expected=round(expected_scope(proposal, wd), 1),
                              groups_list=[dict(flags=k[0], members=list(k[1:]), files=len(v))
                                           for k, v in sorted(proposal.items(), key=lambda kv: -len(kv[1]))]))
            report["regroup"]["%s/%s" % (plat, d)]["_cur"] = cur
            report["regroup"]["%s/%s" % (plat, d)]["_prop"] = {"+".join(k[1:]): v for k, v in proposal.items()}
    if release_keys:
        report["release_keys"] = key_reasons(*release_keys)
    return report


def pattern_scopes(report, patterns):
    """{pattern: {plat/set: dict(current=(groups, files), proposed=(groups, files))}}."""
    out = {}
    for name, files in patterns.items():
        row = {}
        for k, r in report["regroup"].items():
            row[k] = dict(current=scope(r["_cur"], set(files)), proposed=scope(r["_prop"], set(files)),
                          of=(r["current"]["groups"], r["proposed"]["groups"]))
        out[name] = row
    return out


def group_plan(repo, old, new, plat):
    """One line per merged group of PLAT's release build: what a cache built at
    OLD serves at NEW under closure-v2. A group whose closure, members and root
    are unchanged is `hit` wherever OLD's archive of it exists for the same
    flags, toolchain and build box; otherwise `compile: <reason>`. Static: two
    `git archive` trees, no build, no cache listing."""
    t_new = bincache.revision_tree(repo, new)
    t_old = bincache.revision_tree(repo, old)
    lines = []
    old_rows = {(d, stem): (cls, mem) for d, stem, cls, mem in group_rows(t_old, plat)}
    for d, stem, cls, mem in group_rows(t_new, plat):
        new_files = {}
        for m in mem:
            new_files.update(closure_of(t_new, [t_new / "bindings" / (m + ".mojo")]))
        if (d, stem) not in old_rows:
            lines.append("%s %s/%s: compile: new group" % (plat, d, stem))
            continue
        ocls, omem = old_rows[(d, stem)]
        if (ocls, omem) != (cls, mem):
            lines.append("%s %s/%s: compile: membership changed" % (plat, d, stem))
            continue
        old_files = {}
        for m in omem:
            old_files.update(closure_of(t_old, [t_old / "bindings" / (m + ".mojo")]))
        changed = diff_closures(old_files, new_files)
        root_old = merged_bindings.root_text(stem, omem, repo=t_old)
        root_new = merged_bindings.root_text(stem, mem, repo=t_new)
        inputs_old = bincache.executed_inputs(t_old, t_old / "bindings" / "merged_build.sh", merged=True)[0]
        inputs_new = bincache.executed_inputs(t_new, t_new / "bindings" / "merged_build.sh", merged=True)[0]
        why = []
        if changed:
            why.append("closure changed: " + ", ".join(changed[:6]) + (" (+%d)" % (len(changed) - 6)
                                                                        if len(changed) > 6 else ""))
        if root_old != root_new:
            why.append("root text changed")
        if inputs_old != inputs_new:
            why.append("build inputs changed: " + ", ".join(sorted(r for r in set(inputs_old) | set(inputs_new)
                                                                  if inputs_old.get(r) != inputs_new.get(r))))
        lines.append("%s %s/%s: %s" % (plat, d, stem, "compile: " + "; ".join(why) if why
                                       else "hit (closure unchanged since %s)" % old[:12]))
    return lines


def render_tables(report):
    """The generated half of docs/build/COMPILE_CLOSURES.md, from a report JSON."""
    out = []
    w = out.append
    old, new = report.get("old", "")[:10], report.get("new", "")[:10]
    w("### Merged groups at %s: what each compile reads\n" % new)
    w("Closure = distinct Mojo source files the group's members import (closure-v2 walk, `-I . -I bindings`), "
      "generated sources counted as their generator. The v1 key hashed every one of the tree's "
      "%d `.mojo` files for every merged group (its compile line names a generated root; no root was "
      "readable, so the fallback fired)." % report["v1_merged"]["tree_files"])
    w("")
    w("| platform | set | group | flags | members | closure files | changed %s..%s |" % (old, new))
    w("|---|---|---|---|---:|---:|---|")
    for plat, pr in report["platforms"].items():
        for g in pr["groups"]:
            ch = g.get("changed_since_old")
            chs = "-" if ch is None else ("%d: %s" % (len(ch), ", ".join("`%s`" % c for c in ch[:3])) +
                                          (" ..." if len(ch) > 3 else "") if ch else "0 (unchanged)")
            w("| %s | %s | `%s` | %s | %d | %d | %s |" % (plat, g["set"], g["stem"], g["flags"], len(g["members"]),
                                                          g["files"], chs))
    w("")
    sc = report["scripts"]
    sizes = sorted(v["files"] for v in sc.values() if v["files"])
    bad = {k: v for k, v in sc.items() if not v["files"]}
    w("### Per-binding build scripts\n")
    w("%d per-binding scripts under `bindings/` (`build_defines.sh`, `build_host_family.sh` and `merged_build.sh` "
      "are not one binding's compile). Closure sizes: min %d, median %d, max %d files. Unresolved imports: %s. "
      "The only script whose compile line `script_plan` cannot read is `bindings/merged_build.sh` (the "
      "generated root `$root`), which is why every merged key fell back to the whole tree under v1."
      % (len(sc), sizes[0], sizes[len(sizes) // 2], sizes[-1],
         "none" if not bad else ", ".join("%s (%s)" % (k, v["unresolved"] or v["note"]) for k, v in bad.items())))
    w("")
    for plat, pr in report["platforms"].items():
        w("### Files every %s group reads (%d groups): %d\n" % (plat, pr["n_groups"], len(pr["shared_by_all"])))
        w(", ".join("`%s`" % f for f in pr["shared_by_all"]) or "none")
        w("")
    w("### Files ranked by the groups they reach (Linux, 9 groups; top 30)\n")
    w("| file | groups | commits since %s |" % report["since"])
    w("|---|---:|---:|")
    wt = {f: n for n, f in report["weights"]["top"]}
    for f, c in report["platforms"]["linux"]["ranking"][:30]:
        w("| `%s` | %d | %s |" % (f, c, wt.get(f, "")))
    w("")
    w("### Hot shared files: commits since %s x Linux groups reached (top 15)\n" % report["since"])
    w("%d commits since %s touched a `.mojo` file. Each row is how many group rebuilds that file alone "
      "caused on one Linux leg." % (report["weights"]["commits_with_mojo"], report["since"]))
    w("")
    w("| file | commits | groups | group rebuilds |")
    w("|---|---:|---:|---:|")
    for cost, n, c, f in report["hot"][:15]:
        w("| `%s` | %d | %d | %d |" % (f, n, c, cost))
    w("")
    if report.get("release_keys"):
        w("### Why every macOS key changed between the two releases (from their keys/<key>.json)\n")
        w("| group (args) | v1 fields that differ | v1 scope (files) |")
        w("|---|---|---|")
        for args, r in report["release_keys"].items():
            w("| `%s` | %s | %s (%s -> %s) |" % (args, ", ".join("`%s`" % c for c in r["changed"]),
                                                r["new_scope"], r["old_files"], r["new_files"]))
        w("")
    w("### Regrouping by closure overlap (proposal)\n")
    w("Start from one group per member and merge, each step, the pair that saves the most shared compile work "
      "per unit of added incremental scope, until the whole set reads no more files than today; never above "
      "the largest current group (the peak-memory proxy); macOS flags classes stay apart. `expected` is the "
      "commit-weighted mean number of closure files recompiled per changed file since %s." % report["since"])
    w("")
    w("| platform/set | members | groups now -> proposed | full-build files now -> proposed | largest group | expected files per change now -> proposed |")
    w("|---|---:|---|---|---|---|")
    for k, r in report["regroup"].items():
        c, q = r["current"], r["proposed"]
        w("| %s | %d | %d -> %d | %d -> %d | %d -> %d | %.0f -> %.0f (%+.0f%%) |" % (
            k, r["members"], c["groups"], q["groups"], c["full_files"], q["full_files"], c["max_files"],
            q["max_files"], c["expected"], q["expected"], 100.0 * (q["expected"] - c["expected"]) / c["expected"]))
    w("")
    for k, r in report["regroup"].items():
        if not k.startswith("linux/"):
            continue
        w("Proposed %s groups:\n" % k)
        for g in r["proposed"]["groups_list"]:
            w("- %d files: %s" % (g["files"], ", ".join("`%s`" % m for m in g["members"])))
        w("")
    if report.get("patterns"):
        w("### Compile scope of single-file changes: groups (closure files) recompiled, now -> proposed\n")
        keys = list(next(iter(report["patterns"].values())))
        w("| change | " + " | ".join(keys) + " |")
        w("|---|" + "---|" * len(keys))
        for name, row in report["patterns"].items():
            cells = []
            for k in keys:
                cur, prop, of = row[k]["current"], row[k]["proposed"], row[k]["of"]
                cells.append("%d/%d (%d) -> %d/%d (%d)" % (cur[0], of[0], cur[1], prop[0], of[1], prop[1]))
            w("| %s | %s |" % (name, " | ".join(cells)))
        w("")
    return "\n".join(out) + "\n"


BEGIN, END = "<!-- generated by tools/compile_closures.py render: begin -->", \
    "<!-- generated by tools/compile_closures.py render: end -->"


def render_into(doc, report):
    text = Path(doc).read_text() if Path(doc).is_file() else BEGIN + "\n" + END + "\n"
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    Path(doc).write_text(head + BEGIN + "\n\n" + render_tables(report) + "\n" + END + tail)


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [strip_private(v) for v in obj]
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--new", default="HEAD")
    r.add_argument("--old", default="")
    r.add_argument("--since", default=(datetime.date.today() - datetime.timedelta(days=31)).isoformat())
    r.add_argument("--release-keys", nargs=2, metavar=("OLD_KEYS", "NEW_KEYS"))
    r.add_argument("--pattern", action="append", default=[], metavar="NAME=path[,path]")
    r.add_argument("--json", required=True)
    rd = sub.add_parser("render", help="write the generated tables of a report JSON into the doc")
    rd.add_argument("--json", required=True)
    rd.add_argument("--doc", default=str(ROOT / "docs" / "build" / "COMPILE_CLOSURES.md"))
    pl = sub.add_parser("plan", help="per merged group: hit (unchanged since OLD) or compile: <reason>")
    pl.add_argument("--old", required=True)
    pl.add_argument("--new", default="HEAD")
    pl.add_argument("--platform", choices=PLATFORMS, action="append")
    args = ap.parse_args(argv)
    if args.cmd == "render":
        render_into(args.doc, json.loads(Path(args.json).read_text()))
        print("compile_closures: tables written into %s" % args.doc)
        return 0
    if args.cmd == "plan":
        for plat in args.platform or PLATFORMS:
            for line in group_plan(ROOT, args.old, args.new, plat):
                print(line)
        return 0
    rev = lambda x: subprocess.run(["git", "-C", str(ROOT), "rev-parse", x], capture_output=True, text=True,
                                   check=True).stdout.strip() if x else ""
    report = analyse(ROOT, rev(args.old), rev(args.new), args.since, args.release_keys)
    pats = dict(p.split("=", 1) for p in args.pattern)
    report["patterns"] = pattern_scopes(report, {k: v.split(",") for k, v in pats.items()})
    Path(args.json).write_text(json.dumps(strip_private(report), indent=1, sort_keys=True))
    print("compile_closures: %s (%d member closures)" % (args.json, len(report["member_files"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
