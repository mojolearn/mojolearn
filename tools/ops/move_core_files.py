#!/usr/bin/env python3
"""tools/ops/move_core_files.py -- move the single-family files out of core/ and rewrite every reference.

Reads the map (tools/ops/core_move_map.tsv: file, action, destination, ...; rows with action `move` or `delete`
are acted on, `stay` rows are documentation), renames the files with git, and rewrites the references to them:

  mojo-import        `from core.NAME import ...` and `import core.NAME` in every .mojo file (the only two forms of
                     a core import in the repo; there is no relative and no bare-sibling import of a core file)
  mojo-comment       `core/NAME.mojo` in the `#` comment tail of a Mojo line
  patch-header       `diff --git`, `---`, `+++`, `rename from/to` lines of every .patch/.diff (the sabotage arms)
  patch-body         the same two Mojo rewrites applied to the context/added/removed lines of a .mojo hunk, so a
                     patch still applies to the file the first two rewrites produced (line counts never change)
  text-path          `core/NAME.mojo` in every other text file: build scripts, bindings/*.json group lists,
                     tools/, packaging/, .github/, pixi.toml, python/, docs/build/, experiments/ grid controls
  text-dotted        `core.NAME` (a module string) in the same files

Never edited (old paths stay as written; docs/build/CORE_MOVE_2026-11-28.md holds the old-to-new table):
docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md, experiments/removed/, experiments/tombstones/,
verification/records/, docs/rules/RULES_HISTORY.md, CHANGELOG.md, any line holding an inline TOMBSTONE or
`Restore:` marker, and dated notes under docs/ outside docs/build/ and docs-site/ (--docs-all includes them).
Mojo docstrings and string literals are not edited either: a string literal is compiled in, and this move
changes no generated code. What is left is counted and listed, never silently skipped.

  python3 tools/ops/move_core_files.py --dry-run     counts per kind, everything it could not rewrite, the checks
  python3 tools/ops/move_core_files.py               the same, then the edits, `git mv` and `git rm`

Checks, all made on the planned tree before one byte is written (exit 2 and no write if any fails):
  * every import that resolved before the move resolves after it (the resolution of tools/bincache.py resolve():
    roots `.`, `bindings` and the importer's directory). An unresolved import would also widen every binding's
    source closure to the whole tree (bincache.source_digest), so every family would rebuild for ever;
  * no destination exists already, no two rows share a destination, no deleted file is still imported or holds
    a TOMBSTONE marker;
  * with bindings/release_families.json present: every `move` row is still reached by exactly one family from
    the bindings' entry files (other lanes edit importers; a row that became shared is STALE, exit 4,
    --allow-stale to proceed).
After the edits (not in --dry-run): `git apply --check` of every patch it touched; a patch that applied before
and does not apply after is an unresolved reference (exit 3).

Idempotent: a row whose file is already at its destination is skipped, and the reference rewrite of a finished
row finds nothing. Parallel: files are read, rewritten and written by one process per core (the worker count is
the cores this process may use; a worker holds a few MB). The `git mv` step is serial by nature: git has one
index and one lock; it is one call per destination directory, about thirty calls.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = "tools/ops/core_move_map.tsv"
MOVE_DOC = "docs/build/CORE_MOVE_2026-11-28.md"

#: Never edited. A prefix ends with "/"; anything else is one file.
PROTECTED = ("docs/TOMBSTONES.md", "docs/apple-fast/EXPERIMENTS.md", "experiments/removed/",
             "experiments/tombstones/", "verification/records/", "docs/rules/RULES_HISTORY.md", "CHANGELOG.md",
             MOVE_DOC, DEFAULT_MAP, "tools/ops/move_core_files.py",
             # a build list recorded from the 0.8.57 release, compared by its test against that release
             "tools/test_release_host_build_tokenizer_0857.json")
#: Under docs/ only these are rewritten by default (the rest are dated notes).
DOCS_LIVE = ("docs/build/", "docs-site/")
MARKERS = ("TOMBSTONE", "Restore:")

#: bincache.EXTERNAL_TOP is read from tools/bincache.py when it can be imported; this is the fallback.
EXTERNAL_TOP = {"std", "max", "python", "builtin", "memory", "gpu", "sys", "os", "collections", "math", "testing",
                "time", "random", "algorithm", "utils", "bit", "complex", "pathlib", "subprocess", "tempfile",
                "hashlib", "benchmark", "logger", "runtime", "compile", "iter", "io", "format", "layout", "linalg",
                "nn", "extensibility"}

FROM_RE = re.compile(r"^\s*from\s+(\.*[A-Za-z_][\w.]*|\.+)\s+import\s+(.*)$")
IMPORT_RE = re.compile(
    r"^\s*import\s+([A-Za-z_][\w.]*(?:\s+as\s+\w+)?(?:\s*,\s*[A-Za-z_][\w.]*(?:\s+as\s+\w+)?)*)\s*$")
HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
#: An include path that names core (`-I core`, `-I ./core`). The build scripts pass `-I .` and `-I bindings` only.
INCLUDE_RE = re.compile(r"-I\s*\"?(?:\./)?core(?![\w.])")

_STATE = {}


def cores():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def git(*args, check=True, text=True):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), check=check, capture_output=True, text=text)


def read_map(path):
    """([(old, new)], [deleted]) from the map; `stay` rows are ignored."""
    moves, deletes = [], []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0] == "file":
            continue
        if len(cols) < 3:
            sys.exit("%s:%d: fewer than three columns" % (path, n))
        old, action, new = cols[0], cols[1], cols[2]
        if action == "move":
            if not old.startswith("core/") or not old.endswith(".mojo") or not new.endswith(".mojo"):
                sys.exit("%s:%d: a move row must be core/<...>.mojo -> <dir>/<...>.mojo" % (path, n))
            moves.append((old, new))
        elif action == "delete":
            deletes.append(old)
        elif action != "stay":
            sys.exit("%s:%d: unknown action %r" % (path, n, action))
    return moves, deletes


def module_of(path):
    return path[:-len(".mojo")].replace("/", ".")


class Rewriter:
    """The line-local rewrites. One instance per worker, built from the map."""

    def __init__(self, moves):
        self.path = dict(moves)
        self.mod = {module_of(o): module_of(n) for o, n in moves}
        names = sorted((re.escape(o[len("core/"):-len(".mojo")]) for o, _ in moves), key=len, reverse=True)
        alt = "|".join(names) if names else r"(?!x)x"
        dotted = "|".join(n.replace("/", r"\.") for n in names) if names else r"(?!x)x"
        self.imp_from = re.compile(r"^(\s*from\s+)core\.(%s)(\s+import\b)" % dotted)
        self.imp_plain = re.compile(r"^(\s*import\s+)core\.(%s)(?![\w.])" % dotted)
        self.path_re = re.compile(r"(?<![A-Za-z0-9_.\-])core/(%s)\.mojo(?![A-Za-z0-9_])" % alt)
        self.dotted_re = re.compile(r"(?<![A-Za-z0-9_./\-])core\.(%s)(?![A-Za-z0-9_])" % dotted)

    def _path_sub(self, m):
        return self.path["core/" + m.group(1) + ".mojo"]

    def _dot_sub(self, m):
        return self.mod["core." + m.group(1)]

    def mojo_line(self, line, counts, kind_import="mojo-import", kind_comment="mojo-comment"):
        """Import statements, then `core/NAME.mojo` in the comment tail. Never inside a quoted string."""
        if "core" not in line:
            return line
        new, n = self.imp_from.subn(lambda m: m.group(1) + self.mod["core." + m.group(2)] + m.group(3), line)
        if not n:
            new, n = self.imp_plain.subn(lambda m: m.group(1) + self.mod["core." + m.group(2)], line)
        if n:
            counts[kind_import] = counts.get(kind_import, 0) + n
            line = new
        if "#" in line and not any(mark in line for mark in MARKERS):
            cut = comment_start(line)
            if cut >= 0:
                tail, k = self.path_re.subn(self._path_sub, line[cut:])
                if k:
                    counts[kind_comment] = counts.get(kind_comment, 0) + k
                    line = line[:cut] + tail
        return line

    def text_line(self, line, counts, kind_path="text-path", kind_dotted="text-dotted"):
        if "core" not in line or any(mark in line for mark in MARKERS):
            return line
        line, k = self.path_re.subn(self._path_sub, line)
        if k:
            counts[kind_path] = counts.get(kind_path, 0) + k
        line, k = self.dotted_re.subn(self._dot_sub, line)
        if k:
            counts[kind_dotted] = counts.get(kind_dotted, 0) + k
        return line

    def left(self, line):
        """How many references to a moved file the line still holds."""
        if "core" not in line:
            return 0
        return len(self.path_re.findall(line)) + len(self.dotted_re.findall(line))


def comment_start(line):
    """Index of the `#` that starts the comment tail, or -1. A `#` inside a quoted string does not count."""
    quote = None
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#":
            return i
        i += 1
    return -1


def parse_imports(text):
    """(module, [names]) per import statement outside a docstring: tools/bincache.py parse_imports, verbatim in
    behaviour, so this tool and the release's source closure read the same statements."""
    out = []
    in_doc = False
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        n_quotes = line.count('"""')
        if in_doc:
            if n_quotes % 2 == 1:
                in_doc = False
            continue
        if n_quotes % 2 == 1:
            in_doc = True
            continue
        code = line.split("#", 1)[0]
        m = FROM_RE.match(code)
        if m:
            names = m.group(2).strip()
            if names.startswith("(") and ")" not in names:
                while i < len(lines) and ")" not in names:
                    names += " " + lines[i].split("#", 1)[0]
                    i += 1
            names = names.strip("() \t")
            out.append((m.group(1), [n.split()[0] for n in names.split(",") if n.strip()]))
            continue
        m = IMPORT_RE.match(code)
        if m:
            for part in m.group(1).split(","):
                out.append((part.split()[0], []))
    return out


def rewrite_mojo(rw, text, counts):
    lines = text.split("\n")
    return "\n".join(rw.mojo_line(line, counts) for line in lines)


def rewrite_text(rw, text, counts):
    return "\n".join(rw.text_line(line, counts) for line in text.split("\n"))


def rewrite_patch(rw, text, counts):
    """Headers by path; the body of a hunk by the rewrite of the file it patches. Hunk line counts are read from
    the `@@` header, so a removed line that starts with `--` is never taken for a file header."""
    out = []
    old_left = new_left = 0
    target_mojo = False
    for line in text.split("\n"):
        if old_left > 0 or new_left > 0:
            tag = line[:1]
            if tag == "\\":
                out.append(line)
                continue
            if tag == "-":
                old_left -= 1
            elif tag == "+":
                new_left -= 1
            else:
                old_left -= 1
                new_left -= 1
            body = line[1:]
            if target_mojo:
                body = rw.mojo_line(body, counts, "patch-body-import", "patch-body-comment")
            else:
                body = rw.text_line(body, counts, "patch-body-path", "patch-body-dotted")
            out.append(tag + body)
            continue
        m = HUNK_RE.match(line)
        if m:
            old_left = int(m.group(1)) if m.group(1) is not None else 1
            new_left = int(m.group(2)) if m.group(2) is not None else 1
            out.append(line)
            continue
        if line.startswith(("diff --git ", "--- ", "+++ ", "rename from ", "rename to ", "Index: ")):
            new, k = rw.path_re.subn(rw._path_sub, line)
            if k:
                counts["patch-header"] = counts.get("patch-header", 0) + k
            if line.startswith("+++ "):
                target_mojo = new.split("\t")[0].rstrip().endswith(".mojo")
            out.append(new)
            continue
        out.append(line)
    return "\n".join(out)


def protected(rel):
    return any(rel == p or (p.endswith("/") and rel.startswith(p)) for p in PROTECTED)


def category(rel):
    """Where a text reference lives, for the per-kind counts of --dry-run."""
    if rel.endswith(".mojo"):
        return "mojo"
    if rel.endswith((".patch", ".diff")):
        return "patch"
    if rel.startswith("bindings/") and rel.endswith(".json"):
        return "bindings-group-json"
    if rel.endswith(".sh") or rel.startswith("bindings/"):
        return "build-script"
    if rel.startswith(".github/"):
        return "github-workflow"
    if rel.startswith("packaging/"):
        return "packaging"
    if rel.startswith("tools/"):
        return "tools"
    if rel.startswith("python/"):
        return "python-package"
    if rel.startswith("experiments/"):
        return "experiments"
    if rel.startswith(("docs/", "docs-site/")):
        return "docs"
    if rel.startswith("verification/"):
        return "verification"
    return "other:" + (rel.split("/")[0] if "/" in rel else rel)


def _init(moves, deletes, docs_all, write):
    _STATE["rw"] = Rewriter(moves)
    _STATE["del"] = Rewriter([(d, d) for d in deletes if d.startswith("core/") and d.endswith(".mojo")])
    _STATE["docs_all"] = docs_all
    _STATE["write"] = write


def _work(rels):
    """One batch of files: (rel, counts, changed, imports_before, imports_after, left, skipped_reason)."""
    rw, dl = _STATE["rw"], _STATE["del"]
    res = []
    for rel in rels:
        path = ROOT / rel
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        is_mojo = rel.endswith(".mojo")
        if b"core" not in raw and not is_mojo:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            if b"core/" in raw:
                res.append((rel, {}, False, None, None, [], "binary"))
            continue
        counts, left, why = {}, [], None
        imports_before = parse_imports(text) if is_mojo else None
        if "core" not in text:
            res.append((rel, counts, False, imports_before, None, left, why))
            continue
        frozen = protected(rel) or (rel.startswith("docs/") and not rel.startswith(DOCS_LIVE)
                                    and not _STATE["docs_all"])
        if frozen:
            new = text
            why = "permanent record" if protected(rel) else "dated note under docs/"
        elif is_mojo:
            new = rewrite_mojo(rw, text, counts)
        elif rel.endswith((".patch", ".diff")):
            new = rewrite_patch(rw, text, counts)
        else:
            new = rewrite_text(rw, text, counts)
        scan_includes = rel.endswith(".sh") or rel.startswith(".github/") or rel == "pixi.toml"
        for n, line in enumerate(new.split("\n"), 1):
            k = rw.left(line) + dl.left(line)
            if k:
                # (line number, text, the line is a permanent marker, the line names a deleted file)
                left.append((n, line.strip()[:160], any(mark in line for mark in MARKERS), dl.left(line) > 0))
            if scan_includes and "-I" in line and INCLUDE_RE.search(line):
                left.append((n, "INCLUDE " + line.strip()[:150], False, False))
        changed = new != text
        if changed and _STATE["write"]:
            path.write_bytes(new.encode("utf-8"))
        res.append((rel, counts, changed, imports_before, parse_imports(new) if is_mojo and changed else None,
                    left, why))
    return res


def resolves(module, names, importer, files, dirs, external):
    """True when the import names at least one file of the tree, or an external package: bincache.resolve()."""
    idir = importer.rsplit("/", 1)[0] + "/" if "/" in importer else ""
    if module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        base = idir
        for _ in range(dots - 1):
            base = base.rstrip("/").rsplit("/", 1)[0] + "/" if "/" in base.rstrip("/") else ""
        parts = [p for p in module.lstrip(".").split(".") if p]
        search, ext = [base], False
    else:
        parts = module.split(".")
        search, ext = ["", "bindings/", idir], parts[0] in external
    for root in search:
        for k in range(1, len(parts) + 1):
            if root + "/".join(parts[:k]) + "/__init__.mojo" in files:
                return True
        if parts and root + "/".join(parts) + ".mojo" in files:
            return True
        pkg = root + "/".join(parts) + ("/" if parts else "")
        for name in names:
            if name == "*":
                if pkg.rstrip("/") in dirs:
                    return True
                continue
            if pkg + name + ".mojo" in files or pkg + name + "/__init__.mojo" in files:
                return True
    return ext


def targets(module, names, importer, files):
    """The files of the tree an import names (for the family reach)."""
    idir = importer.rsplit("/", 1)[0] + "/" if "/" in importer else ""
    if module.startswith("."):
        return []
    parts = module.split(".")
    got = []
    for root in ("", "bindings/", idir):
        leaf = root + "/".join(parts) + ".mojo"
        if leaf in files:
            got.append(leaf)
        for name in names:
            cand = root + "/".join(parts) + "/" + name + ".mojo"
            if cand in files:
                got.append(cand)
    return got


def family_reach(imports, files, families):
    """{file: set of families} over the import graph from each binding's entry file."""
    adj = {}
    for rel, imps in imports.items():
        seen = set()
        for module, names in imps:
            seen.update(targets(module, names, rel, files))
        adj[rel] = seen
    reach = {}
    for family, bindings in families.items():
        done = set()
        todo = ["bindings/entry/entry_%s.mojo" % b for b in bindings]
        while todo:
            f = todo.pop()
            if f in done or f not in files:
                continue
            done.add(f)
            todo.extend(adj.get(f, ()))
        for f in done:
            reach.setdefault(f, set()).add(family)
    return reach


def show(title, rows, limit):
    print("%s: %d" % (title, len(rows)))
    for row in rows[:limit]:
        print("  " + row)
    if len(rows) > limit:
        print("  ... %d more (--report FILE writes all of them)" % (len(rows) - limit))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=0, help="workers; default every core this process may use")
    ap.add_argument("--docs-all", action="store_true", help="also rewrite dated notes under docs/")
    ap.add_argument("--allow-stale", action="store_true", help="proceed with move rows that are no longer one-family")
    ap.add_argument("--skip-missing", action="store_true", help="drop rows whose file exists at neither path")
    ap.add_argument("--report", help="write every list in full to this file (JSON)")
    ap.add_argument("--limit", type=int, default=40, help="lines printed per list")
    args = ap.parse_args()

    moves, deletes = read_map(ROOT / args.map)
    tracked = [p for p in git("ls-files", "-z").stdout.split("\0") if p]
    untracked = [p for p in git("ls-files", "-z", "-o", "--exclude-standard").stdout.split("\0") if p]
    present = set(p for p in tracked + untracked if (ROOT / p).is_file())
    problems = []

    pending, done, missing = [], [], []
    for old, new in moves:
        if old in present and new in present:
            problems.append("destination exists already: %s -> %s" % (old, new))
        elif old in present:
            pending.append((old, new))
        elif new in present:
            done.append((old, new))
        else:
            missing.append("%s -> %s" % (old, new))
    seen_dest = {}
    for old, new in moves:
        if new in seen_dest:
            problems.append("two rows share the destination %s: %s and %s" % (new, seen_dest[new], old))
        seen_dest[new] = old
    if missing and not args.skip_missing:
        problems.extend("the file exists at neither path (another lane moved or deleted it; --skip-missing): " + m
                        for m in missing)
    pending_del = [d for d in deletes if d in present]
    for d in pending_del:
        if any(mark in (ROOT / d).read_text(errors="replace") for mark in MARKERS):
            problems.append("a deleted file holds a TOMBSTONE marker, which is permanent: " + d)
    other_core = sorted({p.split("/core/")[0] + "/core/" for p in tracked if "/core/" in p})
    if other_core:
        print("NOTE: other directories named core exist (%s); `x/core/NAME.mojo` there would be rewritten too"
              % ", ".join(other_core[:5]))

    workers = args.jobs or cores()
    files = sorted(present)
    batch = max(1, min(64, len(files) // (workers * 8) or 1))
    batches = [files[i:i + batch] for i in range(0, len(files), batch)]

    # Pass 1, always without writing: the plan, in memory.
    def scan(write):
        results = []
        if workers == 1:
            _init(moves, deletes, args.docs_all, write)
            for b in batches:
                results.extend(_work(b))
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                     initargs=(moves, deletes, args.docs_all, write)) as pool:
                for part in pool.map(_work, batches):
                    results.extend(part)
        return results

    results = scan(False)
    totals, by_cat, changed_files, binary = {}, {}, [], []
    left_mojo, left_other, frozen_rows, deleted_refs, include_paths = [], [], [], [], []
    imports_before, imports_after = {}, {}
    rename = dict(pending)
    for rel, counts, changed, before, after, left, why in results:
        if why == "binary":
            binary.append(rel)
            continue
        cat = category(rel)
        for kind, n in counts.items():
            totals[kind] = totals.get(kind, 0) + n
            key = kind if cat in ("mojo", "patch") else "%s [%s]" % (kind, cat)
            by_cat[key] = by_cat.get(key, 0) + n
        if changed:
            changed_files.append(rel)
        if before is not None:
            imports_before[rel] = before
            imports_after[rename.get(rel, rel)] = after if after is not None else before
        for n, line, marker, names_deleted in left:
            row = "%s:%d: %s" % (rel, n, line)
            if line.startswith("INCLUDE "):
                include_paths.append(row)
            elif why or marker:
                # a permanent record, a dated note, or an inline TOMBSTONE / Restore line: old paths stay
                frozen_rows.append(row)
            elif names_deleted:
                deleted_refs.append(row)
            elif rel.endswith(".mojo"):
                # a docstring or a string literal (imports and comment tails were rewritten)
                left_mojo.append(row)
            else:
                left_other.append(row)

    try:
        sys.path.insert(0, str(ROOT / "tools"))
        import bincache
        external = set(bincache.EXTERNAL_TOP)
    except Exception:
        external = set(EXTERNAL_TOP)

    def tree(paths):
        mojo = {p for p in paths if p.endswith(".mojo")}
        return mojo, {p.rsplit("/", 1)[0] for p in mojo if "/" in p}

    files_before, dirs_before = tree(present)
    after_set = (present - set(rename) - set(pending_del)) | set(rename.values())
    files_after, dirs_after = tree(after_set)
    unres_before = {(rel, module) for rel, imps in imports_before.items() for module, names in imps
                    if not resolves(module, names, rel, files_before, dirs_before, external)}
    back = {new: old for old, new in pending}
    unresolved, preexisting = [], 0
    for rel, imps in imports_after.items():
        if rel in pending_del:
            continue
        for module, names in imps:
            if resolves(module, names, rel, files_after, dirs_after, external):
                continue
            if (back.get(rel, rel), module) in unres_before:
                preexisting += 1
                continue
            unresolved.append("%s: from %s import %s" % (rel, module, ", ".join(names[:4])))
    unresolved.sort()

    for rows in (left_mojo, left_other, frozen_rows, deleted_refs, include_paths):
        rows.sort()

    shadows = []
    beside = dirs_after | {p[:-len(".mojo")] for p in files_after}
    for old, new in pending:
        dest = new.rsplit("/", 1)[0]
        tops = {m.split(".")[0] for m, _ in imports_before.get(old, ()) if not m.startswith(".")}
        for top in sorted(tops):
            if top not in external and (dest + "/" + top) in beside:
                shadows.append("%s imports top-level `%s`; %s/%s exists beside it" % (new, top, dest, top))

    stale = []
    fam_path = ROOT / "bindings/release_families.json"
    if fam_path.is_file():
        families = json.loads(fam_path.read_text()).get("families", {})
        reach = family_reach(imports_before, files_before, families)
        for old, new in pending:
            fams = sorted(reach.get(old, ()))
            if len(fams) > 1:
                stale.append("%s is now reached by %s" % (old, ",".join(fams)))
    else:
        print("NOTE: bindings/release_families.json is absent; the one-family check of the map was not made")

    patches = [rel for rel in changed_files if rel.endswith((".patch", ".diff"))]

    print("map: %d move rows (%d pending, %d already moved, %d missing), %d delete rows (%d pending)"
          % (len(moves), len(pending), len(done), len(missing), len(deletes), len(pending_del)))
    print("workers: %d of %d cores; files scanned: %d; files to edit: %d" % (workers, cores(), len(files),
                                                                            len(changed_files)))
    print("references rewritten, by kind:")
    for kind in sorted(totals):
        print("  %-22s %d" % (kind, totals[kind]))
    print("references rewritten, by place:")
    for key in sorted(by_cat):
        print("  %-48s %d" % (key, by_cat[key]))
    print("imports unresolved before the move and unchanged by it (not this tool's): %d" % preexisting)
    show("UNRESOLVED IMPORTS after the move", unresolved, args.limit)
    show("REFERENCES NOT REWRITTEN (outside Mojo sources)", left_other, args.limit)
    show("references to DELETED files", deleted_refs, args.limit)
    show("STALE map rows (no longer one family)", stale, args.limit)
    show("`-I` include paths that name core", include_paths, args.limit)
    show("left as written: Mojo docstrings and string literals naming a moved file", left_mojo, min(args.limit, 15))
    show("left as written: permanent records, dated notes and TOMBSTONE lines (files)",
         sorted({r.split(":")[0] for r in frozen_rows}), min(args.limit, 15))
    show("name beside a destination that a moved file's import could also mean", shadows, min(args.limit, 10))
    show("binary files that contain `core/` (not read)", binary, 10)
    show("patches to re-check with `git apply --check` after the move", patches, min(args.limit, 10))
    show("problems with the map", problems, args.limit)

    if args.report:
        Path(args.report).write_text(json.dumps(dict(
            pending=pending, done=done, missing=missing, deletes=pending_del, totals=totals, by_place=by_cat,
            changed_files=changed_files, unresolved_imports=unresolved, not_rewritten=left_other,
            deleted_refs=deleted_refs, stale=stale, include_paths=include_paths, left_mojo=left_mojo,
            left_frozen=frozen_rows, shadows=shadows, binary=binary, patches=patches,
            problems=problems), indent=1))

    hard = len(unresolved) + len(left_other) + len(deleted_refs) + len(include_paths) + len(problems)
    print("unresolved references: %d" % hard)
    if hard:
        print("REFUSED: nothing was written." if not args.dry_run else "DRY RUN: would refuse.")
        return 2
    if stale and not args.allow_stale:
        print("REFUSED: stale map rows (--allow-stale)." if not args.dry_run else "DRY RUN: would refuse (stale rows).")
        return 4
    if args.dry_run:
        print("DRY RUN: nothing was written.")
        return 0

    def applies(rel):
        return rel, git("apply", "--check", rel, check=False).returncode == 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        applied_before = dict(pool.map(applies, patches))

    scan(True)
    by_dest = {}
    for old, new in pending:
        by_dest.setdefault(new.rsplit("/", 1)[0], []).append((old, new))
    for dest in sorted(by_dest):
        (ROOT / dest).mkdir(parents=True, exist_ok=True)
        same = [old for old, new in by_dest[dest] if old.rsplit("/", 1)[1] == new.rsplit("/", 1)[1]]
        if same:
            git("mv", "--", *same, dest + "/")
        for old, new in by_dest[dest]:
            if old not in same:
                git("mv", "--", old, new)
    if pending_del:
        git("rm", "-q", "--", *pending_del)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        applied_after = dict(pool.map(applies, patches))
    broken = sorted(rel for rel in patches if applied_before.get(rel) and not applied_after.get(rel))
    never = sorted(rel for rel in patches if not applied_before.get(rel))
    show("patches that did not apply before the move either (not this tool's)", never, 10)
    show("PATCHES BROKEN BY THE MOVE", broken, args.limit)
    print("moved %d files, deleted %d, edited %d; review with `git status --short | head` and commit."
          % (len(pending), len(pending_del), len(changed_files)))
    return 3 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
