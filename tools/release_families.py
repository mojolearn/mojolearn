#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A RELEASE REBUILDS AND RE-RECORDS ONLY THE FAMILIES THAT CHANGED (AGENTS.md part 4, Andrew 2026-11-28).

    python3 tools/release_families.py check [--lanes-file F] [--loads]   every binding and lane has one family
    python3 tools/release_families.py derive [--write]                   the lanes' families, from what they load
    python3 tools/release_families.py plan [--commit HEAD] [--json OUT]  the per-family REUSE/BUILD table
    python3 tools/release_families.py loaders [--bindings a,b]            per binding, the lanes that load it
    python3 tools/release_families.py lanes --plan PLAN.json [--lanes-file F | --families a,b]
                                                                         the lanes a plan records, one csv line

THE MAP is bindings/release_families.json: every binding the release ships belongs to exactly one family, plus
`shared`, and every verify lane to THE SET of families it loads (one family, or a list of several; `shared` for a lane
whose loads are unknown). `check` refuses a binding or lane with no family (and, with --loads, a lane that loads a
binding outside its set and shared), a merged file of bindings/merged_groups.json that holds
bindings of two families (merged_span_problems), and a Python glue file with no family or with a family another
family's code imports (python_problems).

THE PYTHON GLUE (`python.rules` in the map) is in a family's identity too: no binary reads a Python file, so a family
whose Python files changed since the previous release keeps its binaries and RECORDS its lanes (python_glue, apply).
Verification tooling is not product glue and is in no family (the map's `_doc` says where the line is).

THE DECISION is code, from the binding identities tools/release_reuse.py computes (never an agent's reading of a
diff). `apply` turns its per-binding REUSE/BUILD rows into whole families:
  * a family BUILDS when any of its bindings has a changed identity, on any target or set (the reason names the
    binding and what changed: closure files, flags, toolchain, builders);
  * a family that shares a merged file (bindings/merged_groups.json: one `mojo build`, one .so) with a BUILD family
    builds too, because that file's bytes are rebuilt and no longer the bytes its columns were recorded from;
  * `shared` builds when the Linux runtime closure (.libs) builds;
  * --rebuild-all builds every family;
  * every row of a BUILD family becomes BUILD; a REUSE family ships the previous release's bytes.
A family RECORDS on every device when it builds and when `shared` builds (every family loads it). A LANE records when
any family of its set records (lanes_to_record): a lane that loads linear and trees keeps its columns through a
release that changed only neural. `shared`'s own lanes (unknown loads) record whenever any family records.
`settle` then refuses to keep a family whose recorded columns do not rest on the bytes shipped (`kept_refusals`): every cell of every lane of a kept
family must come from a record that ran installed release wheels, taken at a commit where each of the family's
bindings had today's identity. Anything else records.

Nothing here builds, rents or records. Tests: tools/test_release_reuse_families.py.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
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
#: the `python.rules` family of a file that is not product glue (verification tooling, tests, documents)
PY_NOT_GLUE = "-"
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
        #: lane -> ONE family (a string; `shared`: its loads are unknown, it records whenever any family does) or
        #: THE SET of families it loads (a list of two or more, none of them `shared`)
        self.lanes = dict(doc.get("lanes") or {})
        self.rules = [tuple(r) for r in (doc.get("lane_rules") or [])]
        for lane, value in list(self.lanes.items()) + [(p, f) for p, f in self.rules]:
            if not isinstance(value, (str, list, tuple)) or not all(isinstance(f, str) for f in _as_set(value)):
                raise SystemExit("release_families: %s must name a family or a list of families, not %r"
                                 % (lane, value))
            for fam in _as_set(value):
                if fam not in self.families:
                    raise SystemExit("release_families: %s names the unknown family %r" % (lane, fam))
        self.metal1_suffix = metal1_suffix
        #: bindings/merged_groups.json as loaded with the map (load, load_at); None: apply reads the tooling's
        self.merged = None
        #: THE PYTHON GLUE: `python.rules` are [pattern, family] over the tracked files under `python.root`
        py = doc.get("python") or {}
        self.py_root = str(py.get("root") or "python/mojolearn").strip("/")
        self.py_rules = [tuple(r) for r in (py.get("rules") or [])]

    def names(self):
        """The families, shared first, the rest in map order."""
        return [SHARED] + [f for f in self.families if f != SHARED]

    def binding(self, name):
        fam = self.by_binding.get(name)
        if fam is None and self.metal1_suffix and name.endswith(self.metal1_suffix):
            fam = self.by_binding.get(name[:-len(self.metal1_suffix)])
        return fam

    def lane_families(self, name):
        """THE SET of families the lane NAME loads, as a sorted tuple: one family, or several; `(shared,)` for a
        lane whose loads are unknown (it records whenever any family records); None for a lane the map does not
        have. A list that is empty or names `shared` is read as unknown, the safe side."""
        value = self.lanes.get(name)
        if value is None:
            for pattern, fam in self.rules:
                if fnmatch.fnmatchcase(name, pattern):
                    value = fam
                    break
        if value is None:
            return None
        fams = tuple(sorted(set(_as_set(value))))
        return fams if fams and SHARED not in fams else (SHARED,)

    def lane(self, name):
        """The lane's one family; `shared` for a lane of several families or of unknown loads (read alone, that
        is the safe answer: record on any change). `lane_families` is what the plan decides from."""
        fams = self.lane_families(name)
        if fams is None:
            return None
        return fams[0] if len(fams) == 1 else SHARED

    def python(self, rel):
        """The family of the file REL (a path under python.root) by the first rule that matches it: a family,
        PY_NOT_GLUE for a file that is not product glue, None when no rule matches."""
        for pattern, fam in self.py_rules:
            if fnmatch.fnmatchcase(rel, pattern):
                return fam
        return None


def _as_set(value):
    """A lane's map value as a list of family names: a string is one family, a list is itself."""
    return [value] if isinstance(value, str) else list(value)


def load(path=None, merged_path=None):
    merged = read_json(merged_path or MERGED) or {}
    fam = Families(read_json(path or MAP) or {}, merged.get("darwin_metal1_suffix") or "_m1")
    fam.merged = merged
    return fam


def load_at(commit, root=ROOT):
    """The map and the merged groups AT COMMIT (`git show`, never a working tree), as the plan of that commit reads
    them; the tooling's own copies when the commit predates the map."""
    def show(rel):
        p = subprocess.run(["git", "-C", str(root), "show", "%s:%s" % (commit, rel)], capture_output=True, text=True)
        if p.returncode != 0:
            return None
        try:
            return json.loads(p.stdout)
        except ValueError:
            return None
    doc, merged = show("bindings/release_families.json"), show("bindings/merged_groups.json")
    if doc is None:
        return load()
    merged = merged if merged is not None else (read_json(MERGED) or {})
    fam = Families(doc, merged.get("darwin_metal1_suffix") or "_m1")
    fam.merged = merged
    return fam


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
    lane that loads a binding outside its set of families and shared (a lane mapped to `shared`, whose loads are
    unknown, may load any). That is the dangerous direction: such a lane would keep its columns when the family
    it loads and is not mapped to changes."""
    problems = ["binding %s has no family (bindings/release_families.json)" % b for b in bindings
                if fam.binding(b) is None]
    problems += ["lane %s has no family (bindings/release_families.json; tools/release_families.py derive --write)"
                 % lane for lane in lanes if fam.lane_families(lane) is None]
    for lane, loaded in sorted((lane_bindings or {}).items()):
        own = fam.lane_families(lane)
        if own is None or SHARED in own:
            continue
        outside = sorted(b for b in loaded if fam.binding(b) not in own + (SHARED,))
        if outside:
            problems.append("lane %s is in family %s but loads %s" % (
                lane, "+".join(own), ", ".join("%s (%s)" % (b, fam.binding(b)) for b in outside[:6])))
    return problems


def lane_loads(lanes=None):
    """{lane: [binding]} each lane loads, from tools/lane_select.py lane_sources: the bindings its doors name, the
    declared host routes, AND the bindings its calls reach (lane_select.lane_called_bindings: the runtime owners a
    lane runs through the shared host owner's import aliases, `trees_native_call`, `labels_native_call`, ...,
    which no door names). The names alone left nine runtime owners with no lane at all, so a lane could keep its
    columns through a rebuild of an owner it runs (lane lane-loads-2, 2026-11-28); the union only adds bindings.
    The expensive derivation (minutes, memoized on disk by lane_select)."""
    import lane_select
    _, why = lane_select.lane_sources()
    called = lane_select.lane_called_bindings()
    out = {}
    for lane, w in why.items():
        if lanes is None or lane in lanes:
            out[lane] = sorted(set(w.get("bindings") or ()) | set(w.get("declared") or ())
                               | set(called.get(lane) or ()))
    return out


def derive(fam, loads):
    """({lane: family or [families]}, [problem]): THE SET of non-shared families of what each lane loads: the
    one family as a string, several as a sorted list. `shared` (record whenever any family records) only for a
    lane whose set is not known: it loads no family's binding at all (nothing ties it to a family, so nothing
    says which change it is safe from), or it loads a binding with no family, which is a problem, never guessed."""
    lanes, problems = {}, []
    for lane, loaded in sorted(loads.items()):
        unknown = [b for b in loaded if fam.binding(b) is None]
        if unknown:
            problems.append("lane %s loads binding(s) with no family: %s" % (lane, ", ".join(unknown)))
        own = sorted({fam.binding(b) for b in loaded} - {None, SHARED})
        if unknown or not own:
            lanes[lane] = SHARED
        else:
            lanes[lane] = own[0] if len(own) == 1 else own
    return lanes, problems


def derive_table(fam, lanes):
    """The lines `derive` prints about a {lane: family or [families]} map: the lanes by the number of families
    they load, and per family the lanes that record when ONLY that family changed (the lanes whose set holds it,
    plus the `shared` lanes, which record on any change). The second column is what the family split saves."""
    sets = {lane: tuple(sorted(set(_as_set(v)))) for lane, v in lanes.items()}
    unknown = sorted(lane for lane, s in sets.items() if SHARED in s or not s)
    sizes = {}
    for lane, s in sets.items():
        if lane not in unknown:
            sizes[len(s)] = sizes.get(len(s), 0) + 1
    out = ["DERIVED-SETS %d lane(s): %s; shared (unknown loads, record on any change) %d" % (
        len(sets), ", ".join("%d famil%s %d" % (n, "y" if n == 1 else "ies", c) for n, c in sorted(sizes.items()))
        or "none with a family", len(unknown))]
    for name in fam.names():
        if name == SHARED:
            continue
        own = sum(1 for lane, s in sets.items() if lane not in unknown and name in s)
        out.append("DERIVED-FAMILY %-10s records %4d of %d lane(s) when it alone changes (%d load it, %d shared)" % (
            name, own + len(unknown), len(sets), own, len(unknown)))
    return out


def loaders(fam, loads, bindings):
    """What moving each of BINDINGS into a family of its own would do to the lanes, from what the lanes load (the
    `loads` of lane_loads): [dict(binding, family, lanes, alone, by_family, mixed)] where `lanes` load the binding,
    `alone` of them load no binding of any family but shared and the binding's own, `by_family` counts those that
    load exactly one OTHER family besides, and `mixed` those that load several. In a family of its own (`meta`) the
    binding makes every `by_family` lane a `shared` lane under derive's one-family rule (it then records whenever
    any family records), and its `alone` lanes are all that the new family would hold."""
    rows = []
    for b in bindings:
        own = fam.binding(b)
        row = dict(binding=b, family=own, lanes=0, alone=0, by_family={}, mixed=0)
        for lane, loaded in sorted(loads.items()):
            if b not in loaded:
                continue
            row["lanes"] += 1
            others = {fam.binding(x) for x in loaded if x != b} - {None, SHARED, own}
            if not others:
                row["alone"] += 1
            elif len(others) == 1:
                name = next(iter(others))
                row["by_family"][name] = row["by_family"].get(name, 0) + 1
            else:
                row["mixed"] += 1
        rows.append(row)
    return rows


# ---------------------------------------------------------------- one family per merged file
def merged_span_problems(fam, merged=None):
    """[problem] for every merged file of bindings/merged_groups.json whose members belong to two families.

    A merged file is one `mojo build` and one .so: when it holds two families, a build of either rebuilds the other's
    bytes and `apply` must build both, so the family rule saves nothing. On macOS a group is written as <stem> and
    <stem>_m1 (tools/merged_bindings.py plan); both are subsets of the group checked here. A member with no family is
    `check`'s report, not this one's."""
    problems = []
    for key, members in merged_files(merged if merged is not None else getattr(fam, "merged", None)):
        here = {}
        for m in members:
            here.setdefault(fam.binding(m), []).append(m)
        here.pop(None, None)
        if len(here) > 1:
            problems.append("merged file %s holds bindings of %d families (%s): one family per merged file "
                            "(bindings/merged_groups.json)" % (key, len(here), "; ".join(
                                "%s: %s" % (f, ", ".join(sorted(ms))) for f, ms in sorted(here.items()))))
    return problems


# ---------------------------------------------------------------- the Python glue of a family
_PY_FROM = re.compile(r"^[ \t]*from[ \t]+(\.+[\w.]*|mojolearn[\w.]*)[ \t]+import[ \t]+(\([^)]*\)|[^\n#]+)", re.M)
_PY_IMPORT = re.compile(r"^[ \t]*import[ \t]+mojolearn\.([\w.]+)", re.M)
_PY_BINDING = re.compile(r"_mojolearn(?:_[a-z0-9]+)*")


def python_tree(commit, root=ROOT, py_root="python/mojolearn"):
    """{path under py_root: blob sha} of every file tracked there at COMMIT: one `git ls-tree -r`, no file is read."""
    p = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "-z", commit, "--", py_root],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise ValueError("git ls-tree %s -- %s: %s" % (commit, py_root, (p.stderr or "").strip()[-200:]))
    out = {}
    for ent in p.stdout.split("\0"):
        meta, _, path = ent.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob" and path.startswith(py_root + "/"):
            out[path[len(py_root) + 1:]] = parts[2]
    return out


def python_owner(fam, rel):
    """The family whose identity a file under python.root is in: its rule's family, SHARED for a file no rule
    matches (the safe side; `check` refuses such a file), None for a file that is not product glue."""
    own = fam.python(rel)
    if own is None or (own != PY_NOT_GLUE and own not in fam.families):
        return SHARED
    return None if own == PY_NOT_GLUE else own


def python_digests(fam, tree):
    """{family: sha256} of each family's Python glue in a python_tree (path and blob of every file, sorted)."""
    lines = {name: [] for name in fam.names()}
    for rel, blob in sorted(tree.items()):
        own = python_owner(fam, rel)
        if own is not None:
            lines[own].append("%s\0%s\n" % (rel, blob))
    return {name: hashlib.sha256("".join(rows).encode()).hexdigest() for name, rows in lines.items()}


def python_glue(fam, commit, prev_commit, root=ROOT):
    """The Python glue facts `apply` takes: dict(changed={family: [files]}, digests={family: sha256}).

    `changed` names, per family, the glue files added, removed or edited between PREV_COMMIT (the source commit of the
    release whose bytes are reused) and COMMIT, both read with the map of COMMIT. With no previous commit, or one this
    clone cannot read, the glue is not known to be unchanged: `shared` is marked changed, so every lane records."""
    now = python_tree(commit, root, fam.py_root)
    out = dict(changed={}, digests=python_digests(fam, now))
    try:
        if not prev_commit:
            raise ValueError("the previous release names no source commit")
        before = python_tree(prev_commit, root, fam.py_root)
    except ValueError as exc:
        out["changed"][SHARED] = ["<unknown: %s>" % str(exc)[:160]]
        return out
    for rel in sorted(set(now) | set(before)):
        if now.get(rel) != before.get(rel):
            own = python_owner(fam, rel)
            if own is not None:
                out["changed"].setdefault(own, []).append(rel)
    return out


def _python_scan(path):
    """(import candidates, binding names) of one Python file, by text: the modules its `from ... import` and
    `import mojolearn...` statements can name, as (dots, dotted module, [imported names]), and every
    `_mojolearn_*` word. Runs in a worker process (python_problems)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], []
    found = []
    for mod, names in _PY_FROM.findall(text):
        dots = len(mod) - len(mod.lstrip("."))
        mod = mod.lstrip(".")
        if not dots:
            mod = mod[len("mojolearn"):].lstrip(".")
        picked = [n.split(" as ")[0].strip() for n in names.strip("()").replace("\n", " ").split(",")]
        found.append((dots, mod, [n for n in picked if n.isidentifier()]))
    found += [(0, mod, []) for mod in _PY_IMPORT.findall(text)]
    return found, sorted(set(_PY_BINDING.findall(text)))


def python_problems(fam, root=ROOT, files=None, workers=None):
    """[problem] of the Python glue map over the tracked files under python.root of ROOT's working tree:

      * a file no rule matches (a runtime file with no family is refused, like a binding);
      * a rule that names a family the map does not have;
      * a file given a family (not `shared`) that a file of another family or of `shared` imports: a lane of that
        other family then runs it, and would keep its columns when the file changes;
      * a file given a family that names a binding of another family.

    Verification tooling (`-`) may import anything. The imports are read as text (`from . import x`,
    `from .x import y`, `from mojolearn.x import y`, `import mojolearn.x`); a module loaded by a name in a string is
    not seen, which is why a file gets a family only when its public names each load a binding of that family.

    One unit per glue file, scanned across the cores (workers = cpu count, capped by the file count; a scan holds
    one small source file in memory)."""
    root = Path(root)
    base = root / fam.py_root
    if files is None:
        p = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", fam.py_root], capture_output=True, text=True)
        files = [f[len(fam.py_root) + 1:] for f in p.stdout.split("\0") if f.startswith(fam.py_root + "/")]
    files = sorted(files)
    known = set(files)
    problems = ["python rule %s names the unknown family %r" % (pattern, own) for pattern, own in fam.py_rules
                if own != PY_NOT_GLUE and own not in fam.families]
    problems += ["python file %s/%s has no family (bindings/release_families.json python.rules)" % (fam.py_root, rel)
                 for rel in files if fam.python(rel) is None]
    glue = [rel for rel in files if rel.endswith(".py") and fam.python(rel) not in (None, PY_NOT_GLUE)]
    workers = max(1, min(len(glue), int(workers or os.cpu_count() or 1)))
    paths = [str(base / rel) for rel in glue]
    if workers == 1:
        scans = [_python_scan(p) for p in paths]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scans = list(pool.map(_python_scan, paths, chunksize=max(1, len(paths) // (workers * 4))))
    for rel, (imports, words) in zip(glue, scans):
        own = fam.python(rel)
        package = rel.split("/")[:-1]
        targets = set()
        for dots, mod, names in imports:
            if dots > len(package) + 1:
                continue
            stem = (package[:len(package) - (dots - 1)] if dots else []) + [p for p in mod.split(".") if p]
            for parts in [stem] + [stem + [n] for n in names]:
                for cand in ("/".join(parts) + ".py", "/".join(parts + ["__init__.py"])):
                    if parts and cand in known and cand != rel:
                        targets.add(cand)
        for target in sorted(targets):
            theirs = fam.python(target)
            if theirs not in (None, PY_NOT_GLUE, SHARED, own):
                problems.append("python glue %s is in family %s but %s (%s) imports it: make it shared or cut the "
                                "import" % (target, theirs, rel, own))
        if own != SHARED:
            other = sorted({"%s (%s)" % (w, fam.binding(w)) for w in words
                            if fam.binding(w) not in (None, SHARED, own)})
            if other:
                problems.append("python glue %s is in family %s but names %s" % (rel, own, ", ".join(other[:6])))
    return problems


# ---------------------------------------------------------------- the family plan
def _new(name):
    return dict(family=name, decision="REUSE", reason="every binding identity unchanged", bindings=0, built=0,
                record=False, record_reason="")


def apply(plan, fam=None, merged=None, rebuild_all=False, python=None):
    """Make the REUSE/BUILD rows of a tools/release_reuse.py plan whole per family (in place) and return the family
    plan, also stored as plan["families"]. SystemExit for a row whose binding has no family.

    PYTHON (python_glue: the glue files that changed since the release whose bytes are reused, and today's digest per
    family): a family whose Python glue changed RECORDS its lanes; its binaries stay REUSE, because no binary reads
    a Python file. Changed `shared` glue records every family. The digests are kept in the plan for kept_refusals."""
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
    files = merged_files(merged if merged is not None else getattr(fam, "merged", None))
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
    glue = {}
    for name, files in sorted(((python or {}).get("changed") or {}).items()):
        # a family the map does not have (a plan of another map) is shared's: the safe side
        glue.setdefault(name if name in fams else SHARED, []).extend(files)

    def named(files):
        return ", ".join(sorted(files)[:4]) + (" and %d more" % (len(files) - 4) if len(files) > 4 else "")

    for name, files in glue.items():
        fams[name]["python_changed"] = sorted(files)
    for f in fams.values():
        if f["decision"] == "BUILD":
            f.update(record=True, record_reason="it builds")
        elif fams[SHARED]["decision"] == "BUILD":
            f.update(record=True, record_reason="shared builds (every family loads it)")
        elif glue.get(SHARED):
            f.update(record=True, record_reason="the shared Python glue changed (%s); every family runs it"
                     % named(glue[SHARED]))
        elif glue.get(f["family"]):
            f.update(record=True, record_reason="its Python glue changed (%s); its binaries are reused"
                     % named(glue[f["family"]]))
        elif f["family"] == SHARED and any_build:
            f.update(record=True, record_reason="its lanes' loads are unknown, and one builds")
    if not fams[SHARED]["record"] and any(f["record"] for f in fams.values()):
        fams[SHARED].update(record=True, record_reason="its lanes' loads are unknown, and one records")
    for r in rows:
        f = fams[r["family"]]
        if f["decision"] == "BUILD" and r["decision"] == "REUSE":
            r.update(decision="BUILD", reason="family %s builds whole: %s" % (f["family"], f["reason"]),
                     reuse_from=None)
    if rebuild_all and rt:
        rt.update(decision="BUILD", reason="--rebuild-all", reuse_from=None)
    out = dict(schema=PLAN_SCHEMA, rebuild_all=bool(rebuild_all), families=[fams[n] for n in fam.names()])
    if (python or {}).get("digests"):
        out["python_digests"] = dict(python["digests"])
    plan["families"] = out
    return out


def by_name(fplan):
    return {f["family"]: f for f in (fplan or {}).get("families") or []}


def lanes_to_record(fplan, lanes, fam=None):
    """The lanes (in the given order) that record: a lane records when ANY family of its set records
    (`Families.lane_families`). A lane mapped to `shared` (unknown loads) records when `shared` records, which
    `apply` and `settle` make true whenever any family records. When the shared core itself changes, `apply`
    marks every family, so every lane of every set records."""
    fam = fam or load()
    rec = {n for n, f in by_name(fplan).items() if f.get("record")}
    unmapped = [lane for lane in lanes if fam.lane_families(lane) is None]
    if unmapped:
        raise SystemExit("release_families: lane(s) with no family: %s (tools/release_families.py derive --write)"
                         % ", ".join(unmapped[:20]))
    return [lane for lane in lanes if rec.intersection(fam.lane_families(lane))]


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
    today's identity at the record's commit, and so must the family's Python glue and the shared glue
    (python_digests at that commit against the plan's). A lane with no cell at all, a record that names no wheels,
    a commit this clone lacks: refused, so the family records.

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
    # A LANE OF SEVERAL FAMILIES IS CHECKED UNDER EACH OF THEM: it keeps its columns only when every family of
    # its set is kept, and each kept one must find the lane's cells resting on its own bindings and glue.
    lanes_of = {n: [lane for lane in lanes if n in (fam.lane_families(lane) or ())] for n in kept}
    cells = _cell_commits(table, [lane for n in kept for lane in lanes_of[n]])
    records = table.get("records") or []
    refusals, digests, glue_at = {}, {}, {}
    host_toolchain = plan.get("host_toolchain")
    # today's Python glue digest per family (apply stores python_glue's); a plan without it checks the bindings only
    today_glue = (plan.get("families") or {}).get("python_digests") or {}
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
                    if why is None and today_glue:
                        if commit not in glue_at:
                            try:
                                glue_at[commit] = python_digests(fam, python_tree(commit, source_root, fam.py_root))
                            except (OSError, ValueError) as exc:
                                glue_at[commit] = "%s: %s" % (type(exc).__name__, exc)
                        then = glue_at[commit]
                        if isinstance(then, str):
                            why = "lane %s: the Python glue at %s is unreadable (%s)" % (lane, commit[:12], then[:160])
                        else:
                            off = [f for f in (n, SHARED) if then.get(f) != today_glue.get(f)]
                            if off:
                                why = "lane %s: its %s cells rest on %s, where the Python glue of %s was different" % (
                                    lane, cls, commit[:12], off[0])
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
        fams[SHARED].update(record=True, record_reason="its lanes' loads are unknown, and one records")
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
    lo = sub.add_parser("loaders", help="per binding, the lanes that load it and the other families they load")
    lo.add_argument("--bindings", default="", help="comma list (default: every _mojolearn_runtime_*_host and "
                                                   "_mojolearn_tokenizer_host)")
    a = ap.parse_args(argv)
    fam = load()
    if a.cmd == "loaders":
        names = [b for b in a.bindings.split(",") if b] or sorted(
            b for b in fam.by_binding if b.startswith("_mojolearn_runtime_") or b == "_mojolearn_tokenizer_host")
        for r in loaders(fam, lane_loads(), names):
            print("LOADERS %-44s family %-9s lanes %4d  alone %4d  mixed %4d  with one other family: %s" % (
                r["binding"], r["family"], r["lanes"], r["alone"], r["mixed"],
                ", ".join("%s %d" % kv for kv in sorted(r["by_family"].items())) or "-"))
        return 0

    def given_lanes(path):
        if not path:
            return all_lanes()
        text = Path(path).read_text()
        return [x for x in text.replace("\n", ",").split(",") if x.strip()]

    if a.cmd == "check":
        lanes = given_lanes(a.lanes_file)
        problems = check(fam, known_bindings(), lanes, lane_loads(set(lanes)) if a.loads else None)
        problems += merged_span_problems(fam)
        problems += python_problems(fam)
        for line in problems:
            print("REFUSED " + line)
        print("FAMILIES-CHECK %s: %d binding(s), %d lane(s), %d merged file(s), %d python rule(s), %d problem(s)" % (
            "ok" if not problems else "REFUSED", len(known_bindings()), len(lanes), len(merged_files(fam.merged)),
            len(fam.py_rules), len(problems)))
        return 0 if not problems else 1
    if a.cmd == "derive":
        lanes, problems = derive(fam, lane_loads())
        counts = {}
        for f in lanes.values():
            key = f if isinstance(f, str) else "+".join(f)
            counts[key] = counts.get(key, 0) + 1
        print("DERIVED %d lane(s): %s" % (len(lanes), ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
        for line in derive_table(fam, lanes):
            print(line)
        for line in problems:
            print("REFUSED " + line)
        if a.write and not problems:
            # NO CATCH-ALL RULE ONCE THE LANES ARE WRITTEN: a lane the map does not name is refused (check,
            # lanes_to_record), never silently kept or silently shared.
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
