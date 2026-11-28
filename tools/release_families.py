#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A RELEASE REBUILDS AND RE-RECORDS ONLY THE FAMILIES THAT CHANGED (AGENTS.md part 4, Andrew 2026-11-28).

    python3 tools/release_families.py check [--lanes-file F] [--loads]   every binding and lane has one family
    python3 tools/release_families.py derive [--write]                   the lanes' families, from what they load
    python3 tools/release_families.py plan [--commit HEAD] [--json OUT]  the per-family REUSE/BUILD table
    python3 tools/release_families.py lanes --plan PLAN.json [--lanes-file F | --families a,b]
                                                                         the lanes a plan records, one csv line

THE MAP is bindings/release_families.json: every binding the release ships and every verify lane belongs to exactly
one family, plus `shared`. `check` refuses a binding or lane with no family (and, with --loads, an explicit lane that
loads a binding of another family than its own and shared).

THE DECISION is code, from the binding identities tools/release_reuse.py computes (never an agent's reading of a
diff). `apply` turns its per-binding REUSE/BUILD rows into whole families:
  * a family BUILDS when any of its bindings has a changed identity, on any target or set (the reason names the
    binding and what changed: closure files, flags, toolchain, builders);
  * a family that shares a merged file (bindings/merged_groups.json: one `mojo build`, one .so) with a BUILD family
    builds too, because that file's bytes are rebuilt and no longer the bytes its columns were recorded from;
  * `shared` builds when the Linux runtime closure (.libs) builds;
  * --rebuild-all builds every family;
  * every row of a BUILD family becomes BUILD; a REUSE family ships the previous release's bytes.
A family RECORDS its lanes on every device when it builds, when `shared` builds (every family loads it), and `shared`'s
own lanes (the lanes that load several families) record whenever any family records. `settle` then refuses to keep a
family whose recorded columns do not rest on the bytes shipped (`kept_refusals`): every cell of every lane of a kept
family must come from a record that ran installed release wheels, taken at a commit where each of the family's
bindings had today's identity. Anything else records.

Nothing here builds, rents or records. Tests: tools/test_release_families.py.
"""
import argparse
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

MAP = ROOT / "bindings" / "release_families.json"
MERGED = ROOT / "bindings" / "merged_groups.json"
SCHEMA = "mojolearn.release-families.v1"
PLAN_SCHEMA = "mojolearn.release-family-plan.v1"
SHARED = "shared"
TABLE = "python/mojolearn/verify_reference/table.json"
RECORDS = "verification/records/identity"
#: identity column class -> the published wheel (release_reuse.previous_release key) its device bytes ship in
CLASS_WHEEL = {"nvidia": "nvidia", "ptx": "nvidia", "amd": "amd", "apple": "macos", "cpu": "linux"}


class Families:
    """bindings/release_families.json, validated: every name in one family, every family named."""

    def __init__(self, doc, metal1_suffix="_m1"):
        if doc.get("schema") != SCHEMA:
            raise SystemExit("release_families: %s is not %s" % (doc.get("schema"), SCHEMA))
        self.doc = doc
        self.families = dict(doc.get("families") or {})
        if SHARED not in self.families:
            raise SystemExit("release_families: the map has no %r family" % SHARED)
        self.by_binding = {}
        for fam, names in self.families.items():
            for name in names:
                if name in self.by_binding:
                    raise SystemExit("release_families: %s is in two families (%s, %s)"
                                     % (name, self.by_binding[name], fam))
                self.by_binding[name] = fam
        self.lanes = dict(doc.get("lanes") or {})
        self.rules = [tuple(r) for r in (doc.get("lane_rules") or [])]
        for lane, fam in list(self.lanes.items()) + [(p, f) for p, f in self.rules]:
            if fam not in self.families:
                raise SystemExit("release_families: %s names the unknown family %r" % (lane, fam))
        self.metal1_suffix = metal1_suffix

    def names(self):
        """The families, shared first, the rest in map order."""
        return [SHARED] + [f for f in self.families if f != SHARED]

    def binding(self, name):
        fam = self.by_binding.get(name)
        if fam is None and self.metal1_suffix and name.endswith(self.metal1_suffix):
            fam = self.by_binding.get(name[:-len(self.metal1_suffix)])
        return fam

    def lane(self, name):
        if name in self.lanes:
            return self.lanes[name]
        for pattern, fam in self.rules:
            if fnmatch.fnmatchcase(name, pattern):
                return fam
        return None


def load(path=None, merged_path=None):
    merged = read_json(merged_path or MERGED) or {}
    return Families(read_json(path or MAP) or {}, merged.get("darwin_metal1_suffix") or "_m1")


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- what must have a family
def merged_files(merged=None):
    """[(dir/stem, [members])] of bindings/merged_groups.json: one entry per physical merged file, with the
    optional members (byte LM) in the group they join."""
    merged = merged if merged is not None else (read_json(MERGED) or {})
    out = []
    for dname, d in sorted((merged.get("dirs") or {}).items()):
        groups = {stem: list(members) for stem, members in (d.get("groups") or {}).items()}
        for name, opt in (d.get("optional") or {}).items():
            if opt.get("group") in groups:
                groups[opt["group"]].append(name)
        out += [("%s/%s" % (dname, stem), members) for stem, members in sorted(groups.items())]
    return out


def known_bindings(root=ROOT):
    """Every binding name the release and the builds know: the development groups, the merged groups' members and
    the release rows of tools/release_reuse.py (both targets, every tier and the host bindings)."""
    names = set()
    dev = read_json(Path(root) / "bindings" / "development_groups.json") or {}
    for tier in (dev.get("groups") or {}).values():
        for members in tier.values():
            names.update(members)
    merged = read_json(Path(root) / "bindings" / "merged_groups.json") or {}
    for _, members in merged_files(merged):
        names.update(members)
    names.update(merged.get("darwin_metal1") or [])
    import release_reuse
    for target in (release_reuse.LINUX, release_reuse.MACOS):
        names.update(b.name for b in release_reuse.bindings(target))
    return sorted(names)


def all_lanes(root=ROOT, python=None):
    """Every verify lane (tools/identity_columns.py lanes --scope all), the list every column records from."""
    p = subprocess.run([python or sys.executable, "tools/identity_columns.py", "lanes", "--scope", "all"],
                       cwd=str(root), capture_output=True, text=True)
    text = (p.stdout or "").strip()
    if p.returncode != 0 or not text:
        raise SystemExit("release_families: could not list the lanes (rc %s): %s"
                         % (p.returncode, (p.stderr or "").strip()[-400:]))
    return [x for x in text.splitlines()[-1].split(",") if x]


def check(fam, bindings, lanes, lane_bindings=None):
    """[problem] for every binding or lane with no family; with lane_bindings ({lane: [binding]}) also every
    EXPLICIT lane that loads a binding outside its family and shared (a rule-mapped lane to `shared` may load any)."""
    problems = ["binding %s has no family (bindings/release_families.json)" % b for b in bindings
                if fam.binding(b) is None]
    problems += ["lane %s has no family (bindings/release_families.json; tools/release_families.py derive --write)"
                 % lane for lane in lanes if fam.lane(lane) is None]
    for lane, loaded in sorted((lane_bindings or {}).items()):
        own = fam.lane(lane)
        if own is None or own == SHARED:
            continue
        outside = sorted(b for b in loaded if fam.binding(b) not in (own, SHARED))
        if outside:
            problems.append("lane %s is in family %s but loads %s" % (
                lane, own, ", ".join("%s (%s)" % (b, fam.binding(b)) for b in outside[:6])))
    return problems


def lane_loads(lanes=None):
    """{lane: [binding]} each lane loads, from tools/lane_select.py lane_sources (its bindings and the declared
    host routes); the expensive derivation (minutes, memoized on disk by lane_select)."""
    import lane_select
    _, why = lane_select.lane_sources()
    out = {}
    for lane, w in why.items():
        if lanes is None or lane in lanes:
            out[lane] = sorted(set(w.get("bindings") or ()) | set(w.get("declared") or ()))
    return out


def derive(fam, loads):
    """({lane: family}, [problem]): the one non-shared family of what each lane loads, else shared (it loads
    none, or several families). A loaded binding with no family is a problem, never guessed."""
    lanes, problems = {}, []
    for lane, loaded in sorted(loads.items()):
        unknown = [b for b in loaded if fam.binding(b) is None]
        if unknown:
            problems.append("lane %s loads binding(s) with no family: %s" % (lane, ", ".join(unknown)))
        own = {fam.binding(b) for b in loaded} - {None, SHARED}
        lanes[lane] = own.pop() if len(own) == 1 else SHARED
    return lanes, problems


# ---------------------------------------------------------------- the family plan
def _new(name):
    return dict(family=name, decision="REUSE", reason="every binding identity unchanged", bindings=0, built=0,
                record=False, record_reason="")


def apply(plan, fam=None, merged=None, rebuild_all=False):
    """Make the REUSE/BUILD rows of a tools/release_reuse.py plan whole per family (in place) and return the family
    plan, also stored as plan["families"]. SystemExit for a row whose binding has no family."""
    fam = fam or load()
    rows = plan["rows"]
    fams = {f: _new(f) for f in fam.names()}
    unmapped = sorted({r["name"] for r in rows if fam.binding(r["name"]) is None})
    if unmapped:
        raise SystemExit("release_families: no family for binding(s) %s (bindings/release_families.json)"
                         % ", ".join(unmapped))
    for r in rows:
        f = fams[fam.binding(r["name"])]
        r["family"] = f["family"]
        f["bindings"] += 1
        if r["decision"] == "BUILD":
            f["built"] += 1
            if f["decision"] == "REUSE":
                f.update(decision="BUILD", reason="%s: %s" % (r["key"], r["reason"]))
    if rebuild_all:
        for f in fams.values():
            f.update(decision="BUILD", reason="--rebuild-all")
    rt = plan.get("runtime") or {}
    if rt.get("decision") == "BUILD" and fams[SHARED]["decision"] == "REUSE":
        fams[SHARED].update(decision="BUILD", reason="runtime .libs: " + str(rt.get("reason")))
    # A MERGED FILE IS ONE BUILD: when one of its members builds, the leg builds the whole file and its other members'
    # bytes are the leg's (release_reuse.place_merged), so their families' recorded columns no longer hold.
    files = merged_files(merged)
    moved = True
    while moved:
        moved = False
        for key, members in files:
            here = sorted({fam.binding(m) for m in members} - {None})
            building = [f for f in here if fams[f]["decision"] == "BUILD"]
            for f in here:
                if building and fams[f]["decision"] == "REUSE":
                    fams[f].update(decision="BUILD", reason="shares the merged file %s with family %s (BUILD)"
                                   % (key, building[0]))
                    moved = True
    any_build = any(f["decision"] == "BUILD" for f in fams.values())
    for f in fams.values():
        if f["decision"] == "BUILD":
            f.update(record=True, record_reason="it builds")
        elif fams[SHARED]["decision"] == "BUILD":
            f.update(record=True, record_reason="shared builds (every family loads it)")
        elif f["family"] == SHARED and any_build:
            f.update(record=True, record_reason="its lanes load several families, and one builds")
    for r in rows:
        f = fams[r["family"]]
        if f["decision"] == "BUILD" and r["decision"] == "REUSE":
            r.update(decision="BUILD", reason="family %s builds whole: %s" % (f["family"], f["reason"]),
                     reuse_from=None)
    if rebuild_all and rt:
        rt.update(decision="BUILD", reason="--rebuild-all", reuse_from=None)
    out = dict(schema=PLAN_SCHEMA, rebuild_all=bool(rebuild_all), families=[fams[n] for n in fam.names()])
    plan["families"] = out
    return out


def by_name(fplan):
    return {f["family"]: f for f in (fplan or {}).get("families") or []}


def lanes_to_record(fplan, lanes, fam=None):
    """The lanes (in the given order) of every family that records."""
    fam = fam or load()
    rec = {n for n, f in by_name(fplan).items() if f.get("record")}
    unmapped = [lane for lane in lanes if fam.lane(lane) is None]
    if unmapped:
        raise SystemExit("release_families: lane(s) with no family: %s (tools/release_families.py derive --write)"
                         % ", ".join(unmapped[:20]))
    return [lane for lane in lanes if fam.lane(lane) in rec]


def recording(fplan):
    return [n for n, f in by_name(fplan).items() if f.get("record")]


def render(fplan):
    fs = (fplan or {}).get("families") or []
    n_b = sum(f.get("decision") == "BUILD" for f in fs)
    out = ["== release families: %d REUSE / %d BUILD; %d record their lanes on every device%s" % (
        len(fs) - n_b, n_b, sum(bool(f.get("record")) for f in fs),
        " (--rebuild-all)" if (fplan or {}).get("rebuild_all") else "")]
    out.append("   %-10s %-6s %-7s %8s  %s" % ("family", "bytes", "lanes", "built", "reason"))
    for f in fs:
        out.append("   %-10s %-6s %-7s %8s  %s%s" % (
            f["family"], f.get("decision", "-"), "RECORD" if f.get("record") else "KEEP",
            "%d/%d" % (f.get("built", 0), f.get("bindings", 0)), str(f.get("reason", ""))[:220],
            "" if not f.get("record") or f.get("decision") == "BUILD"
            else "; records: " + str(f.get("record_reason", ""))[:220]))
    return out


# ---------------------------------------------------------------- a kept family's columns rest on the shipped bytes
def _cell_commits(table, lanes):
    """{lane: {(class, record index)}} of every cell part of each lane in the reference table."""
    out = {lane: set() for lane in lanes}
    for key, parts in (table.get("cells") or {}).items():
        lane = key.split("/", 1)[0]
        if lane not in out:
            continue
        for ent in (parts or {}).values():
            for cls, ref in ((ent or {}).get("cols") or {}).items():
                out[lane].add((cls, ref if isinstance(ref, int) else ref[0]))
    return out


def _family_digests(plan_rows, commit, cache_dir, root, host_toolchain):
    """{row key: identity digest} of plan_rows at COMMIT (release_reuse.identity over that commit's facts)."""
    import release_reuse as RR
    facts = RR.facts_for_commit(commit, cache_dir, root)
    out = {}
    for r in plan_rows:
        b = RR.Binding(r["target"], r["vendor"], r["arch"], r["tier"], r["name"])
        ident = RR.identity(b, facts, host_toolchain)
        out[r["key"]] = RR.digest_of(ident) if ident else None
    return out


def kept_refusals(plan, lanes, fam=None, source_root=ROOT, cache_dir=None):
    """{family: reason} for every family the plan keeps (no RECORD) whose recorded columns do not rest on the bytes
    it ships. Read from the reference table at SOURCE_ROOT and the records it indexes.

    Each cell part of each lane of a kept family names a record (class, commit). The record must exist and have run
    installed release wheels (_record_refusal), and every binding of the family (every target and set) must have
    today's identity at the record's commit. A lane with no cell at all, a record that names no wheels, a commit
    this clone lacks: refused, so the family records.

    The commits are evaluated one after another, each through release_reuse.facts_for_commit, which is cached by
    commit and walks its build scripts' closures across every core (tree_facts); a second process pool over commits
    would nest pools inside it."""
    fam = fam or load()
    source_root = Path(source_root)
    cache_dir = cache_dir or (source_root / ".release-family-facts")
    table = read_json(source_root / TABLE)
    kept = [n for n, f in by_name(plan.get("families")).items() if not f.get("record")]
    if not kept:
        return {}
    if not table:
        return {n: "no reference table at %s" % TABLE for n in kept}
    rows_of = {n: [r for r in plan["rows"] if r.get("family") == n] for n in kept}
    lanes_of = {n: [lane for lane in lanes if fam.lane(lane) == n] for n in kept}
    cells = _cell_commits(table, [lane for n in kept for lane in lanes_of[n]])
    records = table.get("records") or []
    refusals, digests = {}, {}
    host_toolchain = plan.get("host_toolchain")
    for n in kept:
        why = None
        for lane in lanes_of[n]:
            if not cells[lane]:
                why = "lane %s has no recorded cell" % lane
                break
            for cls, idx in sorted(cells[lane]):
                rec = records[idx] if 0 <= idx < len(records) else None
                if rec is None:
                    why = "lane %s: cell names record %s, not in the table" % (lane, idx)
                    break
                why = _record_refusal(rec, cls, source_root)
                if why is None:
                    commit = rec.get("commit") or ""
                    want = {r["key"]: r.get("identity_digest") for r in rows_of[n]}
                    if commit not in digests:
                        try:
                            digests[commit] = _family_digests(
                                [r for k in kept for r in rows_of[k]], commit, cache_dir, source_root, host_toolchain)
                        except (SystemExit, OSError, ValueError, subprocess.CalledProcessError) as exc:
                            digests[commit] = "%s: %s" % (type(exc).__name__, exc)
                    got = digests[commit]
                    if isinstance(got, str):
                        why = "lane %s: the identities at %s are unreadable (%s)" % (lane, commit[:12], got[:160])
                    else:
                        moved = [k for k, d in want.items() if got.get(k) != d]
                        if moved:
                            why = "lane %s: its %s cells rest on %s, where %s had another identity" % (
                                lane, cls, commit[:12], moved[0])
                if why:
                    why = why if why.startswith("lane ") else "lane %s: %s" % (lane, why)
                    break
            if why:
                break
        if why:
            refusals[n] = why
    return refusals


def _record_refusal(rec, cls, source_root):
    """Why one table record cannot stand for the shipped bytes of its class, or None.

    WHY THE INSTALLED WHEELS AND NOT A VERSION OR A WHEEL SHA. A kept family's bytes are the bytes of the release
    that last built it (each later release reused them, sha256-verified by release_reuse), and that release recorded
    its lanes from its installed wheels (a build always records); the newest record wins per cell, so a cell rests on
    that release's record or a later one of the same bytes. Every release's wheels have their own version and sha256
    (the version is in the metadata, the core and macOS wheels are repacked with the table), so neither names the
    bytes; the identity at the record's commit (kept_refusals) and the fact that the record ran installed release
    wheels do. A record that ran a source tree (no run.wheels: development records) never stands for shipped bytes."""
    if CLASS_WHEEL.get(cls) is None:
        return "unknown column class %s" % cls
    path = Path(source_root) / RECORDS / str(rec.get("dir") or "") / str(rec.get("file") or "")
    doc = read_json(path)
    if doc is None:
        return "record %s/%s is not in %s" % (rec.get("dir"), rec.get("file"), RECORDS)
    wheels = [w for w in ((doc.get("run") or {}).get("wheels") or []) if isinstance(w, dict) and w.get("sha256")]
    if not wheels:
        return "record %s/%s names no installed wheels (not taken from a release wheel)" % (rec.get("dir"), rec.get("file"))
    return None


def settle(plan, lanes, fam=None, source_root=ROOT, cache_dir=None):
    """apply's record flags with kept_refusals folded in (in place): a refused kept family records, with the reason.
    Returns the lanes to record."""
    fam = fam or load()
    refusals = kept_refusals(plan, lanes, fam, source_root, cache_dir)
    fams = by_name(plan.get("families"))
    for n, why in refusals.items():
        fams[n].update(record=True, record_reason="its kept columns do not rest on the shipped bytes: " + why)
    if refusals and not fams[SHARED].get("record"):
        fams[SHARED].update(record=True, record_reason="its lanes load several families, and one records")
    plan["families"]["kept_refusals"] = refusals
    return lanes_to_record(plan["families"], lanes, fam)


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--lanes-file", default="", help="a comma or newline list of lanes (default: identity_columns lanes)")
    c.add_argument("--loads", action="store_true", help="also check each explicit lane against what it loads")
    d = sub.add_parser("derive")
    d.add_argument("--write", action="store_true", help="write the lanes into bindings/release_families.json")
    p = sub.add_parser("plan")
    p.add_argument("--commit", default="HEAD")
    p.add_argument("--json", default="")
    p.add_argument("--rebuild-all", action="store_true")
    p.add_argument("--cache-dir", default=str(ROOT / ".release-family-facts"))
    lanes_p = sub.add_parser("lanes")
    lanes_p.add_argument("--plan", default="", help="a plan JSON carrying `families` (release_reuse plan or this plan)")
    lanes_p.add_argument("--families", default="", help="comma list of families to record (`none`: no family), "
                                                         "instead of --plan")
    lanes_p.add_argument("--settle", action="store_true",
                         help="with a release_reuse --plan: also move the kept families whose columns do not rest "
                              "on the shipped bytes to RECORD (kept_refusals, over this tree's reference table)")
    lanes_p.add_argument("--lanes-file", default="")
    a = ap.parse_args(argv)
    fam = load()

    def given_lanes(path):
        if not path:
            return all_lanes()
        text = Path(path).read_text()
        return [x for x in text.replace("\n", ",").split(",") if x.strip()]

    if a.cmd == "check":
        lanes = given_lanes(a.lanes_file)
        problems = check(fam, known_bindings(), lanes, lane_loads(set(lanes)) if a.loads else None)
        for line in problems:
            print("REFUSED " + line)
        print("FAMILIES-CHECK %s: %d binding(s), %d lane(s), %d problem(s)" % (
            "ok" if not problems else "REFUSED", len(known_bindings()), len(lanes), len(problems)))
        return 0 if not problems else 1
    if a.cmd == "derive":
        lanes, problems = derive(fam, lane_loads())
        counts = {}
        for f in lanes.values():
            counts[f] = counts.get(f, 0) + 1
        print("DERIVED %d lane(s): %s" % (len(lanes), ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
        for line in problems:
            print("REFUSED " + line)
        if a.write and not problems:
            doc = dict(fam.doc, lanes=lanes, lane_rules=[])
            MAP.write_text(json.dumps(doc, indent=1) + "\n")
            print("wrote %s" % MAP)
        return 0 if not problems else 1
    if a.cmd == "plan":
        import release_reuse as RR
        plan = RR.make_plan(a.commit, a.cache_dir, rebuild_all=a.rebuild_all)
        print(RR.render(plan))
        if a.json:
            RR.atomic_write_text(a.json, json.dumps(plan, indent=1, sort_keys=True) + "\n")
        return 0
    if a.cmd == "lanes":
        lanes = given_lanes(a.lanes_file)
        if a.families:
            want = [] if a.families == "none" else [f for f in a.families.split(",") if f]
            unknown = [f for f in want if f not in fam.families]
            if unknown:
                raise SystemExit("release_families: unknown famil(y/ies) %s" % ", ".join(unknown))
            fplan = dict(families=[dict(family=f, record=f in want) for f in fam.names()])
        else:
            doc = read_json(a.plan) if a.plan else None
            if not doc:
                raise SystemExit("release_families: lanes needs --families or a readable --plan")
            if a.settle:
                if not doc.get("rows") or not isinstance(doc.get("families"), dict):
                    raise SystemExit("release_families: --settle needs a release_reuse plan with families")
                settle(doc, lanes, fam)
            fplan = doc.get("families") if isinstance(doc.get("families"), dict) else doc
        for line in render(fplan):
            print("FAMILY " + line.strip(), file=sys.stderr)
        print(",".join(lanes_to_record(fplan, lanes, fam)))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
