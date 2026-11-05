#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The experiment-define guard modules: where they are, who imports them, and the permanent-record check.

    python3 tools/experiment_guards.py check     # layout + permanent record against the pre-split file
    python3 tools/experiment_guards.py files     # the guard modules, one per line

The guards (`comptime assert` refusals of illegal or removed `-D MOJOLEARN_*` sets, with their TOMBSTONE
markers) were one file, core/six_lane_experiment_guards.mojo, that every binding read through
checks/numerics.mojo: 203 commits since 2026-10-04 each rebuilt every merged group. Since 2026-11-04 that file
keeps only the guards whose defines are read in code by bindings of more than one family; each family's guards
live in core/experiment_guards_<family>.mojo, imported by that family's bindings and asserted first thing in
their build_<name>() (docs/build/HOT_GROUPS.md). Tools that read "the guard file" read GUARD_FILES.

Build tooling only; never compiles.
"""
import collections
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED = "core/six_lane_experiment_guards.mojo"
FAMILY_PREFIX = "core/experiment_guards_"
#: The commit whose core/six_lane_experiment_guards.mojo was split (integ/tooling-20261104 head).
SPLIT_BASE = "9ee5208756e69c518660c387c84fd54fdc317da4"
IMPORT_RE = re.compile(r"^from core\.experiment_guards_(\w+) import (\w+)_GUARDS_OK$", re.M)


def family_files(root=ROOT):
    return sorted(str(p.relative_to(root)) for p in Path(root).glob(FAMILY_PREFIX + "*.mojo"))


def guard_files(root=ROOT):
    """The shared guard file first, then every family module (sorted)."""
    return [SHARED] + family_files(root)


def is_guard_file(rel):
    rel = str(rel)
    return rel == SHARED or (rel.startswith(FAMILY_PREFIX) and rel.endswith(".mojo") and "/" not in rel[len("core/"):])


def family_of(rel):
    """'neural' for core/experiment_guards_neural.mojo, 'shared' for the shared file, else None."""
    rel = str(rel)
    if rel == SHARED:
        return "shared"
    if is_guard_file(rel):
        return rel[len(FAMILY_PREFIX):-len(".mojo")]
    return None


def guard_texts(root=ROOT):
    """[(rel, text)] in guard_files order."""
    return [(rel, (Path(root) / rel).read_text()) for rel in guard_files(root)]


def body_lines(text):
    """The lines of _check_configuration's body (between its def line and `return True`)."""
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("def _check_configuration"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "return True")
    return lines[start + 1:end]


def binding_families(root=ROOT, names=None):
    """{binding: [families it imports]} over bindings/<name>.mojo."""
    if names is None:
        sys.path.insert(0, str(Path(root) / "tools"))
        import merged_bindings
        names = merged_bindings.binding_names(root)
    out = {}
    for n in names:
        text = (Path(root) / "bindings" / (n + ".mojo")).read_text()
        out[n] = [m.group(1) for m in IMPORT_RE.finditer(text) if m.group(2) == m.group(1).upper()]
    return out


def layout_problems(root=ROOT, names=None):
    """Why the layout is wrong, or []: each guard module is a leaf (std imports only) defining
    _check_configuration and its constant; each binding imports exactly one family module and asserts its
    constant on the first line of build_<name>()."""
    out = []
    fams = {family_of(f) for f in family_files(root)}
    for rel, text in guard_texts(root):
        for line in text.split("\n"):
            if re.match(r"(from|import) ", line) and not re.match(r"(from|import) std\.", line):
                out.append("%s imports %r: a guard module is a leaf (std only)" % (rel, line))
        const = "SIX_LANE_CONFIGURATION_OK" if rel == SHARED else family_of(rel).upper() + "_GUARDS_OK"
        if "comptime %s = _check_configuration()" % const not in text:
            out.append("%s does not define %s" % (rel, const))
        try:
            body_lines(text)
        except StopIteration:
            out.append("%s has no _check_configuration body" % rel)
    for name, imported in binding_families(root, names).items():
        if len(imported) != 1:
            out.append("bindings/%s.mojo imports %d family guard modules %s (exactly one)" % (name, len(imported), imported))
            continue
        fam = imported[0]
        if fam not in fams:
            out.append("bindings/%s.mojo imports core/experiment_guards_%s.mojo, which does not exist" % (name, fam))
        lines = (Path(root) / "bindings" / (name + ".mojo")).read_text().split("\n")
        try:
            d = lines.index("def build_%s() -> PythonObject:" % name)
        except ValueError:
            out.append("bindings/%s.mojo has no build_%s()" % (name, name)); continue
        if not lines[d + 1].startswith("    comptime assert %s_GUARDS_OK, " % fam.upper()):
            out.append("bindings/%s.mojo: build_%s() does not assert %s_GUARDS_OK first" % (name, name, fam.upper()))
    return out


def _is_code(line):
    s = line.strip()
    return bool(s) and not s.startswith("#")


def record_problems(before_text, after_texts):
    """Why the split broke the permanent record, or []. BEFORE_TEXT is the pre-split guard file, AFTER_TEXTS
    {rel: text} the modules now. Every code line of the old body appears exactly once across the modules;
    every comment line (TOMBSTONE, Restore, provenance) appears at least once; inside each module the old
    lines keep their old relative order. Lines added after the split are not judged."""
    old = body_lines(before_text)
    pos = collections.defaultdict(list)
    for i, l in enumerate(old):
        pos[l].append(i)
    want = collections.Counter(l for l in old if _is_code(l))
    have = collections.Counter()
    seen_comments = set()
    out = []
    for rel, text in after_texts.items():
        last = -1
        for l in body_lines(text):
            if l not in pos:
                continue
            if _is_code(l):
                have[l] += 1
            else:
                seen_comments.add(l)
            nxt = [p for p in pos[l] if p > last]
            if not nxt:
                out.append("%s: %r is out of its original order" % (rel, l.strip()[:80]))
            else:
                last = nxt[0] if _is_code(l) else min(nxt)
    for l, n in want.items():
        if have[l] != n:
            out.append("code line %s %d times (was %d): %r" % ("kept" if have[l] else "dropped", have[l], n, l.strip()[:100]))
    for l in old:
        if l.strip() and not _is_code(l) and l not in seen_comments:
            out.append("comment line dropped: %r" % l.strip()[:100])
    return out


def base_text(root=ROOT, rev=SPLIT_BASE):
    """The pre-split guard file from git, or None when the object is not available (shallow clone)."""
    r = subprocess.run(["git", "-C", str(root), "show", "%s:%s" % (rev, SHARED)], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "check"
    if cmd == "files":
        print("\n".join(guard_files()))
        return 0
    if cmd != "check":
        print(__doc__, file=sys.stderr)
        return 2
    problems = layout_problems()
    before = base_text()
    if before is None:
        print("experiment-guards: pre-split %s not in this clone; record check skipped" % SPLIT_BASE[:10])
    else:
        problems += record_problems(before, dict(guard_texts()))
    for p in problems[:50]:
        print("experiment-guards: " + p)
    print("experiment-guards: %s (%d modules, %d problems)" % ("OK" if not problems else "FAILED",
                                                                 len(guard_files()), len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
