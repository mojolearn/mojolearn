#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE RELEASE PLAN: one dry-run decision, with a reason for every scheduled job.

    pixi run release 0.8.66 --plan [--plan-json FILE]      (tools/release.py; runs nothing)
    python3 tools/release_plan.py decide --facts facts.json --json plan.json
    python3 tools/release_plan.py show --json plan.json

`tools/release.py <v> --plan` gathers the facts (the changed paths since the previous release's source
commit, the per-group compile plan of tools/compile_closures.py, the per-column identity selection of
tools/identity_columns.py changed-lanes, each gate's and producer's recorded state, the boxes the box
steps would need) and hands them to `decide`, which applies the DECISION TABLE below and writes
plan.json. Docs: docs/release/RELEASE_PLAN.md. The plan never launches anything: publication gates on
the evidence it names, and `gate-boxes` rents nothing when the plan owes no device work.

DECISION TABLE (the change since the previous release decides the class; every row says why):

  metadata      only docs, changelog, version or tooling files changed: metadata and package checks
                run; every compile group hits, every identity row and numerical gate is reused.
  repackage     packaging or loader files changed, no product source: the unchanged qualified binaries
                are repackaged; artifact byte checks and the changed loading boundary run, nothing else.
  algorithm     product source changed and reaches some compile groups: those groups compile and the
                identity lanes their closure reaches record; everything else is reused.
  shared        product source reaching every group, or the toolchain / build recipe changed: every
                dependent compiles and records, each one justified by the changed file in its closure.

  and, whatever the class:
  restore       a reused item whose evidence is not on this machine but is retained (R2, a retained
                release directory) is restored BEFORE any execution step; with no copy anywhere it runs.
  blocked       a genuine unresolved failure blocks publication; nothing is waived by the plan.
"""
import argparse
import datetime
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "mojolearn.release-plan.v1"
WASTE_SCHEMA = "mojolearn.release-waste.v1"
ROOT = Path(__file__).resolve().parent.parent

#: Path kinds, first match wins. `metadata` never reaches a compile or a lane (0.8.65: the version bump in
#: python/mojolearn/_version.py "reached" five hf lanes and blocked them); `tooling` builds or checks but
#: ships no product byte; `packaging` and `loader` change how unchanged binaries are packed or loaded;
#: `toolchain` changes what every compile reads; `native` and `python` are product source.
KINDS = (
    ("metadata", ("docs/*", "docs-site/*", "*.md", "CHANGELOG*", "LICENSE*", "AUTHORS*", "CITATION*",
                  "python/mojolearn/_version.py", "docs/release/*-waiver.json", ".github/ISSUE_TEMPLATE/*")),
    # pixi.toml/pixi.lock pin the compiler; the build scripts and the group manifest decide every group's
    # recipe (a comment-only script edit is caught per group by `inputs_changed`, the closure-v2 rule).
    ("toolchain", ("pixi.toml", "pixi.lock", "bindings/build_defines.sh", "bindings/merged_build.sh",
                   "bindings/merged_groups.json")),
    ("loader", ("python/mojolearn/_merged.py", "python/mojolearn/_backend.py", "python/mojolearn/gpu_plugins.py",
                "python/mojolearn/_native*.py", "python/mojolearn/__init__.py")),
    ("packaging", ("packaging/*", "python/pyproject.toml", "python/setup.py", "python/MANIFEST.in")),
    ("tooling", ("tools/*", "bench/*", ".github/*", "tests/*", "*/tests/*", "*/test_*.py", "test_*.py",
                 "experiments/removed/*")),
    ("native", ("*.mojo", "bindings/*")),
    ("python", ("python/mojolearn/*",)),
)
PRODUCT = ("native", "python")
#: gate/producer kinds: which decision-table classes run them
GATE_RUNS_IN = {
    "metadata": {"metadata", "repackage", "algorithm", "shared"},
    "package": {"metadata", "repackage", "algorithm", "shared"},
    "artifact": {"repackage", "algorithm", "shared"},
    "loading": {"repackage", "algorithm", "shared"},
    "numerical": {"algorithm", "shared"},
}


def kind(path):
    for name, pats in KINDS:
        if any(fnmatch.fnmatch(path, p) for p in pats):
            return name
    return "python" if path.startswith("python/") else "tooling"


def classify(changes, groups):
    """(class, reasons) of a change set. `groups` is {gid: {closure: [files]}}."""
    kinds = {p: kind(p) for p in changes}
    if not changes:
        return "metadata", ["no path changed since the previous release"]
    toolchain = sorted(p for p, k in kinds.items() if k == "toolchain")
    if toolchain:
        return "shared", ["toolchain or build recipe changed: " + ", ".join(toolchain[:6])]
    product = sorted(p for p, k in kinds.items() if k in PRODUCT)
    if not product:
        boundary = sorted(p for p, k in kinds.items() if k in ("packaging", "loader"))
        if boundary:
            return "repackage", ["packaging/loading only: " + ", ".join(boundary[:6])]
        return "metadata", ["docs, version or tooling only (%d path(s))" % len(changes)]
    reached = {g for g, row in groups.items() if set(row.get("closure") or ()) & set(product)}
    if groups and len(reached) == len(groups):
        widest = max(product, key=lambda p: sum(p in set(r.get("closure") or ()) for r in groups.values()))
        return "shared", ["%s reaches all %d compile groups" % (widest, len(groups))]
    return "algorithm", ["%d product path(s) reach %d of %d compile groups" % (len(product), len(reached), len(groups))]


def invalidation(changes, groups):
    """{changed file: [gids whose closure holds it]} (tools/bincache.py explain prints the same)."""
    out = {}
    for p in changes:
        hit = sorted(g for g, row in groups.items() if p in set(row.get("closure") or ()))
        if hit:
            out[p] = hit
    return out


def compile_rows(cls, changes, groups, macos_paths=None):
    """groups: {gid: dict(closure, new, inputs_changed, evidence)}. inputs_changed is the closure-v2
    comparison of the executed build inputs (None when not computed: then a toolchain path compiles)."""
    rows = []
    product = {p for p in changes if kind(p) in PRODUCT}
    pins = sorted(p for p in changes if p in ("pixi.toml", "pixi.lock"))
    paths = macos_paths or {}
    moved = paths.get("previous") and paths.get("this") and paths["previous"] != paths["this"]
    recipe = sorted(p for p in changes if kind(p) == "toolchain" and p not in pins)
    for gid, row in sorted(groups.items()):
        reach = sorted(product & set(row.get("closure") or ()))
        changed_inputs = row.get("inputs_changed")
        if row.get("new"):
            action, why = "compile", "new group or membership changed (no previous archive)"
        elif pins:
            action, why = "compile", "toolchain pin changed: " + ", ".join(pins)
        elif changed_inputs or (changed_inputs is None and recipe):
            action, why = "compile", "build inputs changed: " + ", ".join(
                (changed_inputs if isinstance(changed_inputs, list) else recipe)[:4])
        elif reach:
            action, why = "compile", "closure changed: " + ", ".join(reach[:6]) + (
                " (+%d)" % (len(reach) - 6) if len(reach) > 6 else "")
        elif moved and gid.startswith("darwin/"):
            # repo_path is a closure-v2 key field (absolute paths are embedded in the binaries): the macOS
            # release must build from one stable checkout path (docs/release/RELEASE_PLAN.md)
            action, why = "compile", "the macOS build path moved: the previous release built at %s, this one at %s" % (
                paths["previous"], paths["this"])
        else:
            action, why = "hit", "closure unchanged since the previous release"
        r = dict(id=gid, section="compile", action=action, reason=why)
        if action == "hit":
            r["evidence"] = row.get("evidence") or {}
        rows.append(r)
    return rows


def identity_rows(cls, lanes, previous):
    """lanes: {column: {lane: dict(owed, reason, since, since_after_previous, record)}}."""
    rows = []
    for col, by_lane in sorted(lanes.items()):
        for lane, f in sorted(by_lane.items()):
            rid = "%s/%s" % (col, lane)
            if not f.get("owed"):
                rows.append(dict(id=rid, section="identity", action="reuse", reason="closure unchanged since its record %s"
                                 % (f.get("since") or "")[:12], evidence=f.get("evidence") or {"local": True}))
            elif cls in ("metadata", "repackage") and f.get("since_after_previous"):
                # The record rests on or after the previous release and only metadata, tooling or
                # packaging changed since then: no numerical rerun (decision table).
                rows.append(dict(id=rid, section="identity", action="reuse", reason="decision table %s: no product source changed "
                                 "since the previous release %s (selector said: %s)"
                                 % (cls, (previous or "")[:12], f.get("reason")),
                                 evidence=f.get("evidence") or {"local": True}))
            else:
                rows.append(dict(id=rid, section="identity", action="record", reason=f.get("reason") or "closure changed",
                                 device=f.get("device")))
    return rows


def gate_rows(cls, gates, compiles_owed=True):
    """gates: {name: dict(kind, state: done|owed|failed|held, reason, device, evidence, previous_evidence,
    self_checked)}. A check the class does not run is reused from the previous release's evidence; an
    `artifact` (build) step with every compile group a hit takes the previous binaries; with no previous
    evidence the check runs (never a reuse without evidence)."""
    rows = []
    for name, g in sorted(gates.items()):
        k = g.get("kind", "numerical")
        runs = cls in GATE_RUNS_IN.get(k, GATE_RUNS_IN["numerical"])
        if g.get("state") == "done":
            rows.append(dict(id=name, section="gates", action="reuse", reason="recorded for this commit: " + (g.get("reason") or ""),
                             evidence=g.get("evidence") or {"local": True}))
        elif g.get("self_checked") and g.get("state") == "owed":
            # A box step decides its own reuse (tools/release.py box_step_owed: a gate adopts a retained
            # verdict for the same bytes itself); the plan never claims a reuse that step would refuse.
            rows.append(dict(id=name, section="gates", action="run", reason="its own check owes work: "
                             + (g.get("reason") or "not recorded"), device=g.get("device")))
        elif not runs and k == "artifact" and not compiles_owed:
            rows.append(dict(id=name, section="gates", action="reuse", reason="decision table %s: every compile "
                             "group hits; the previous release's binaries are taken" % cls,
                             evidence=g.get("previous_evidence") or {"local": True}))
        elif not runs and g.get("previous_evidence"):
            rows.append(dict(id=name, section="gates", action="reuse", reason="decision table %s: a %s check does not run; the "
                             "previous release's evidence covers the same bytes" % (cls, k),
                             evidence=g["previous_evidence"]))
        else:
            rows.append(dict(id=name, section="gates", action="run", reason=g.get("reason") or (
                "%s check owed for a %s change" % (k, cls)), device=g.get("device")))
    return rows


def restore_rows(rows):
    """Reused rows whose evidence is retained but not local: restored before any execution. A reused row
    with no copy anywhere becomes owed work (it is not reuse without evidence)."""
    out = []
    for r in rows:
        if r["action"] not in ("hit", "reuse"):
            continue
        ev = r.get("evidence") or {}
        if ev.get("local", True):
            continue
        if ev.get("retained"):
            out.append(dict(id=r["id"], section="restore", action="restore", reason="evidence is not on this machine; restore it "
                            "from " + ev["retained"] + " before any execution", source=ev["retained"],
                            local_path=ev.get("local_path")))
        else:
            r["action"] = dict(compile="compile", identity="record", gates="run")[r["section"]]
            r["reason"] = "no evidence on this machine and none retained: " + r["reason"]
    return out


def box_rows(work, leases):
    """work: device-owed rows; leases: {vendor: minutes}. Rent only what owed device work needs."""
    need = {}
    for r in work:
        if r.get("device"):
            need.setdefault(r["device"], []).append(r["id"])
    if not need:
        return dict(action="none", reason="no owed device work: gate-boxes rents nothing", boxes={})
    boxes = {v: dict(action="rent", minutes=leases.get(v), for_rows=len(ids), first=ids[:5])
             for v, ids in sorted(need.items())}
    return dict(action="rent", reason="device work owed on " + ", ".join(sorted(need)), boxes=boxes)


def decide(facts):
    """The plan of a release from its facts (pure; tools/test_release_plan.py covers each table row)."""
    changes = sorted(facts.get("changes") or [])
    groups = facts.get("groups") or {}
    cls, why = classify(changes, groups)
    comp = compile_rows(cls, changes, groups, facts.get("macos_paths"))
    ident = identity_rows(cls, facts.get("lanes") or {}, facts.get("previous"))
    gates = gate_rows(cls, facts.get("gates") or {}, any(r["action"] == "compile" for r in comp))
    restore = restore_rows(comp + ident + gates)
    work = [r for r in comp + ident + gates if r["action"] in ("compile", "record", "run")]
    boxes = box_rows(work, facts.get("leases") or {})
    blocked = []
    for f in facts.get("failures") or []:
        if not f.get("resolved"):
            blocked.append(dict(id=f["id"], reason="unresolved failure: " + f.get("reason", "")))
    for r in work:
        blocked.append(dict(id=r["id"], reason="owed: " + r["reason"]))
    for r in restore:
        blocked.append(dict(id=r["id"], reason="evidence to restore first: " + r["source"]))
    plan = dict(schema=SCHEMA, version=facts.get("version"), commit=facts.get("commit"),
                previous=facts.get("previous"), change_class=cls, class_reasons=why,
                changes=[dict(path=p, kind=kind(p)) for p in changes],
                invalidates=invalidation(changes, groups),
                compile=comp, identity=ident, gates=gates, restore=restore, boxes=boxes,
                publish=dict(status="ready" if not blocked else "blocked", blocked=blocked,
                             gated_on=[r["id"] for r in comp + ident + gates]),
                launches_nothing=True)
    plan["waste"] = waste_summary(plan, facts.get("step_seconds") or {}, facts.get("box_wait_seconds"),
                                  facts.get("admission_seconds"), facts.get("group_seconds") or {})
    if facts.get("previous_waste"):
        plan["waste_delta"] = waste_delta(plan["waste"], facts["previous_waste"])
    return plan


# ---------------------------------------------------------------- waste.json
def waste_summary(plan, step_seconds, box_wait=None, admission=None, group_seconds=None):
    """The numbers the plan already has plus step times (docs/release/RELEASE_PLAN.md, waste.json)."""
    comp, ident = plan["compile"], plan["identity"]
    reused = [r["id"] for r in comp if r["action"] == "hit"]
    rebuilt = [r["id"] for r in comp if r["action"] == "compile"]
    gs = group_seconds or {}
    return dict(schema=WASTE_SCHEMA, version=plan.get("version"), commit=plan.get("commit"),
                groups_reused=len(reused), groups_rebuilt=len(rebuilt),
                groups_reused_seconds=round(sum(gs.get(g, 0) for g in reused), 1),
                groups_rebuilt_seconds=round(sum(gs.get(g, 0) for g in rebuilt), 1),
                lanes_reused=sum(r["action"] == "reuse" for r in ident),
                lanes_recorded=sum(r["action"] == "record" for r in ident),
                box_wait_seconds=box_wait, admission_seconds=admission,
                steps={k: v for k, v in sorted(step_seconds.items())})


def waste_delta(now, before):
    keys = ("groups_reused", "groups_rebuilt", "groups_rebuilt_seconds", "lanes_reused", "lanes_recorded",
            "box_wait_seconds", "admission_seconds")
    out = {}
    for k in keys:
        a, b = now.get(k), before.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[k] = round(a - b, 1)
    out["against"] = before.get("version")
    return out


def write_waste(path, waste):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(waste, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def previous_waste(release_root, version):
    """The newest waste.json of an earlier version under RELEASE_ROOT (<version>/<commit12>/waste.json)."""
    def key(v):
        return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.a-z]+", v) if x)
    root = Path(release_root)
    best = None
    for p in root.glob("*/*/waste.json"):
        v = p.parent.parent.name
        if key(v) < key(version) and (best is None or key(v) > key(best[0]) or
                                      (key(v) == key(best[0]) and p.stat().st_mtime > best[1].stat().st_mtime)):
            best = (v, p)
    if best is None:
        return None
    try:
        return json.loads(best[1].read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- output
def lines(plan):
    out = ["PLAN %s at %s (previous %s): class %s -- %s" % (
        plan.get("version"), (plan.get("commit") or "")[:12], (plan.get("previous") or "none")[:12],
        plan["change_class"], "; ".join(plan["class_reasons"]))]
    for path, gids in sorted(plan["invalidates"].items()):
        out.append("  %s invalidates these %d groups: %s" % (path, len(gids), ", ".join(gids)))
    for section in ("restore", "compile", "identity", "gates"):
        rows = plan[section]
        counts = {}
        for r in rows:
            counts[r["action"]] = counts.get(r["action"], 0) + 1
        out.append("%s: %s" % (section, ", ".join("%s %d" % kv for kv in sorted(counts.items())) or "none"))
        for r in rows:
            if section == "identity" and r["action"] == "reuse":
                continue  # hundreds of rows; counted above, listed in plan.json
            out.append("  %-8s %s: %s" % (r["action"], r["id"], r["reason"][:200]))
    b = plan["boxes"]
    out.append("boxes: " + b["action"] + " -- " + b["reason"])
    for v, row in sorted(b["boxes"].items()):
        out.append("  rent %s for %s minutes (%d owed row(s), e.g. %s)" % (
            v, row.get("minutes"), row["for_rows"], ", ".join(row["first"])))
    p = plan["publish"]
    out.append("publish: %s%s" % (p["status"], " (%d row(s))" % len(p["blocked"]) if p["blocked"] else ""))
    for r in p["blocked"][:40]:
        out.append("  blocked %s: %s" % (r["id"], r["reason"][:200]))
    w = plan["waste"]
    out.append("waste: groups reused %d / rebuilt %d (%.0f s), lanes reused %d / recorded %d" % (
        w["groups_reused"], w["groups_rebuilt"], w["groups_rebuilt_seconds"], w["lanes_reused"], w["lanes_recorded"]))
    if plan.get("waste_delta"):
        d = plan["waste_delta"]
        out.append("waste vs %s: %s" % (d.get("against"), ", ".join("%s %+g" % (k, v) for k, v in sorted(d.items())
                                                                   if k != "against")))
    if plan.get("report_totals"):
        # the status report's totals line (tools/release_report.py, docs/release/STATUS_REPORT.md)
        out.append(plan["report_totals"])
    out.append("the plan launches nothing; publication gates on the rows above")
    return out


# ---------------------------------------------------------------- facts from the repository
def git(*args, repo=ROOT):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def changed_paths(previous, commit, repo=ROOT):
    return sorted(x for x in git("diff", "--name-only", previous, commit, repo=repo).splitlines() if x)


def group_facts(repo, previous, commit, platforms=("darwin", "linux")):
    """{plat/set/stem: dict(closure, new, inputs_changed)} at COMMIT: the closure-v2 walk of
    tools/compile_closures.py, membership against PREVIOUS, and the executed build inputs (shell lines
    hashed without comments) compared between the two trees. Static: two `git archive` trees."""
    sys.path.insert(0, str(Path(repo) / "tools"))
    import bincache
    import compile_closures as cc
    t_new = bincache.revision_tree(Path(repo), commit)
    t_old = bincache.revision_tree(Path(repo), previous) if previous else None

    def inputs(tree):
        return bincache.executed_inputs(tree, tree / "bindings" / "merged_build.sh", merged=True)[0]
    moved = None
    if t_old is not None:
        a, b = inputs(t_old), inputs(t_new)
        moved = sorted(r for r in set(a) | set(b) if a.get(r) != b.get(r))
    out = {}
    for plat in platforms:
        old = {(d, s): list(m) for d, s, _c, m in cc.group_rows(t_old, plat)} if t_old else {}
        for d, stem, _cls, mem in cc.group_rows(t_new, plat):
            closure = set()
            for m in mem:
                closure |= set(cc.closure_of(t_new, [t_new / "bindings" / (m + ".mojo")]))
            out["%s/%s/%s" % (plat, d, stem)] = dict(closure=sorted(closure), new=old.get((d, stem)) != list(mem),
                                                    inputs_changed=moved)
    return out


LANE_REASON = re.compile(r"since ([0-9a-f]{7,40})")


def lane_facts(summary, previous, column, device=None, is_ancestor=None):
    """Per-lane facts of one column from an identity_columns.py changed-lanes --summary-json document."""
    out = {}
    for lane, why in (summary.get("record") or {}).items():
        m = LANE_REASON.search(why or "")
        since = m.group(1) if m else None
        after = bool(since and previous and is_ancestor and is_ancestor(previous, since))
        out[lane] = dict(owed=True, reason=why, since=since, since_after_previous=after, device=device)
    for lane, row in (summary.get("reused") or {}).items():
        out[lane] = dict(owed=False, since=(row or {}).get("commit"), device=device)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide", help="facts.json -> plan.json (pure)")
    d.add_argument("--facts", required=True)
    d.add_argument("--json", required=True)
    s = sub.add_parser("show")
    s.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "decide":
        plan = decide(json.loads(Path(args.facts).read_text()))
        Path(args.json).write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    else:
        plan = json.loads(Path(args.json).read_text())
    print("\n".join(lines(plan)))
    return 0 if plan["publish"]["status"] == "ready" else 3


if __name__ == "__main__":
    sys.exit(main())
