#!/usr/bin/env python3
"""Refuse a new runtime `Python.import_module(` call in bindings/ and providers/.

The rule (Andrew, 2026-11-05, lane perf-last-items): a native door reaches a
Python module through the import-time connection (`native_loaded_module(...)`,
bindings/native_loaded_module.mojo), never by calling `Python.import_module`
on a call path, where every call pays the import machinery (a `sys.modules`
lookup through Python, the import lock and attribute traffic). This checker
reports, in every .mojo file under

    bindings/**            the Python extension entry points
    providers/**           the provider shells

each code line that calls `Python.import_module(`, unless the call sits inside
an import-time initializer:

    a `def PyInit_<name>` function (the module's import entry), or
    a line carrying, or directly below a line carrying,
    `# import-time: <what>` (a one-time cold initializer, for example a
    process `_Global` state fill or a module-construction step)

Comment lines and triple-quoted docstrings are skipped. The sites left on the
lane/perf-cpython-table sweep (2026-11-06) are listed in
tools/hooks/runtime_import_module_baseline.txt as `path:line: code`; a hit is
grandfathered by a row with the same path and the same code text (the line
number is informational). The list only shrinks: convert a site to
`native_loaded_module(...)` (connecting the module at import if needed) and
prune its row.

Modes:
    no_runtime_import_module.py --tree [REF] [--branch] [--baseline FILE]
        Scan every bindings/ and providers/ .mojo file committed at REF (default
        HEAD). Fail (exit 1) on any hit not covered by the baseline. The
        baseline is REF's file, restricted to the rows main's baseline also
        has, so a branch cannot grow it. Rows no hit matches are stale.
        --branch (the pre-push hook passes it for every ref except main): judge
        only the files REF adds or changes over merge-base(origin/main, REF); a
        hit that was already in that file at the merge-base is main's debt.
    no_runtime_import_module.py --prune-baseline [REF]
        Rewrite the worktree baseline without its stale rows and with current
        line numbers. It never adds a row.
Exit 0 clean, 1 on findings, 2 on usage errors. Pure git and regex; no Mojo, no build.
"""
import os
import re
import subprocess
import sys
from collections import Counter

BASELINE = "tools/hooks/runtime_import_module_baseline.txt"
CALL = re.compile(r"\bPython\s*\.\s*import_module\s*\(")
TAG = re.compile(r"#\s*import-time:")
DEF = re.compile(r"^(?P<indent>\s*)def\s+(?P<name>\w+)")
ROW = re.compile(r"^(?P<path>[^:\s]+\.mojo):(?P<line>\d+):\s?(?P<code>.*)$")


def is_judged(path):
    """True for the runtime door files this checker judges (see the module docstring)."""
    return path.endswith(".mojo") and (path.startswith("bindings/") or path.startswith("providers/"))


def _code_part(line):
    """The line without its trailing `#` comment (a `#` inside a string literal is kept)."""
    quote = None
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#":
            return line[:i]
        i += 1
    return line


def scan(text):
    """[(line number, code)] of the runtime `Python.import_module(` calls in one file."""
    out = []
    lines = text.splitlines()
    in_doc = None
    # the stack of enclosing defs as (indent width, name); a line belongs to
    # every def whose indent is smaller than its own
    defs = []
    for n, raw in enumerate(lines, 1):
        s = raw.strip()
        if in_doc:
            if in_doc in s:
                in_doc = None
            continue
        q = '"""' if '"""' in s else ("'''" if "'''" in s else None)
        if q:
            # a line with a triple quote is docstring or string text, never a call
            if s.count(q) % 2 == 1:
                in_doc = q
            continue
        if not s or s.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        while defs and defs[-1][0] >= indent:
            defs.pop()
        m = DEF.match(raw)
        if m:
            defs.append((len(m["indent"]), m["name"]))
        code = _code_part(raw)
        if not CALL.search(code):
            continue
        prev = lines[n - 2] if n >= 2 else ""
        if TAG.search(raw) or TAG.search(prev):
            continue
        if any(name.startswith("PyInit_") for _w, name in defs):
            continue
        out.append((n, s))
    return out


def _git(*args, inp=None):
    return subprocess.run(["git", *args], capture_output=True, input=inp)


def _resolve(ref):
    r = _git("rev-parse", "--verify", "--quiet", ref + "^{commit}")
    return r.stdout.decode().strip() if r.returncode == 0 else None


def _blobs(ref):
    """{path: blob sha} of every judged .mojo file at ref."""
    r = _git("ls-tree", "-r", "--full-tree", ref)
    if r.returncode != 0:
        raise SystemExit(f"no_runtime_import_module: cannot list {ref}: {r.stderr.decode().strip()}")
    out = {}
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        meta, _, path = ln.partition("\t")
        bits = meta.split()
        if len(bits) == 3 and bits[1] == "blob" and is_judged(path):
            out[path] = bits[2]
    return out


def _read(shas):
    """{sha: text}, one `git cat-file --batch` process."""
    need = list(dict.fromkeys(shas))
    if not need:
        return {}
    out = _git("cat-file", "--batch", inp="".join(s + "\n" for s in need).encode()).stdout
    texts, i = {}, 0
    for sha in need:
        nl = out.index(b"\n", i)
        head = out[i:nl].split()
        i = nl + 1
        if len(head) < 3 or head[1] == b"missing":
            texts[sha] = ""
            continue
        size = int(head[2])
        texts[sha] = out[i:i + size].decode("utf-8", "replace")
        i += size + 1
    return texts


def hits_at(ref, paths=None):
    """[(path, line, code)] at ref, sorted by path and line (only `paths`, if given)."""
    blobs = _blobs(ref)
    if paths is not None:
        blobs = {p: s for p, s in blobs.items() if p in paths}
    texts = _read(blobs.values())
    return [(p, n, c) for p in sorted(blobs) for n, c in scan(texts[blobs[p]])]


def parse_baseline(text):
    """[(path, line, code)] rows of a baseline file; comments and blank lines skipped."""
    rows = []
    for ln in text.splitlines():
        m = ROW.match(ln.strip())
        if m and not ln.lstrip().startswith("#"):
            rows.append((m["path"], int(m["line"]), m["code"].strip()))
    return rows


def format_row(path, line, code):
    return f"{path}:{line}: {code}"


def _keys(rows):
    return Counter((r[0], r[-1]) for r in rows)


def _baseline_at(ref):
    r = _git("show", f"{ref}:{BASELINE}")
    return parse_baseline(r.stdout.decode("utf-8", "replace")) if r.returncode == 0 else None


def _main_ref():
    for m in ("refs/remotes/origin/main", "origin/main", "refs/heads/main"):
        if _resolve(m):
            return m
    return None


def _per_dir(hits):
    c = Counter(h[0].rsplit("/", 1)[0] if "/" in h[0] else "." for h in hits)
    return ", ".join(f"{d} {n}" for d, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))) or "none"


def check_tree(ref, branch=False, baseline_file=None):
    sha = _resolve(ref)
    if not sha:
        print(f"no_runtime_import_module: unknown ref {ref}", file=sys.stderr)
        return 2
    if baseline_file:
        with open(baseline_file, encoding="utf-8") as f:
            base = _keys(parse_baseline(f.read()))
    else:
        base = _keys(_baseline_at(sha) or [])
    main = _main_ref()
    main_rows = _baseline_at(main) if main else None
    grown = Counter()
    if main_rows is not None:
        main_base = _keys(main_rows)
        grown = base - main_base
        base = base & main_base  # the baseline only shrinks: rows main lacks are not grandfathered
    hits = hits_at(sha)
    judged = hits
    allow = Counter(base)
    if branch and main:
        mb = _git("merge-base", main, sha).stdout.decode().strip()
        if mb and mb != sha:
            changed = set(_git("diff", "--name-only", "--no-renames", mb, sha, "--", "*.mojo").stdout.decode().split())
            judged = [h for h in hits if h[0] in changed]
            old = _keys(hits_at(mb, changed))
            allow = Counter({k: max(base[k], old[k]) for k in set(base) | set(old)})
    seen = Counter()
    new = []
    for h in judged:
        key = (h[0], h[2])
        seen[key] += 1
        if seen[key] > allow[key]:
            new.append(h)
    stale = Counter() if branch else base - _keys(hits)
    print(f"no_runtime_import_module: {len(hits)} runtime Python.import_module calls in {len({h[0] for h in hits})} of "
          f"{len(_blobs(sha))} bindings/providers .mojo files scanned at {sha[:9]} (baseline {sum(base.values())}); "
          f"per directory: {_per_dir(hits)}", file=sys.stderr)
    for (p, c), n in sorted(grown.items()):
        print(f"no_runtime_import_module: baseline row {p}: {c} (x{n}) is not in main's baseline; "
              f"the baseline only shrinks", file=sys.stderr)
    for (p, c), n in sorted(stale.items()):
        print(f"no_runtime_import_module: stale baseline row {p}: {c} (x{n}) matches no hit: "
              f"remove it with --prune-baseline", file=sys.stderr)
    for p, n, c in new:
        print(f"{p}:{n}: runtime Python.import_module call; use native_loaded_module(...) with an import-time "
              f"connection (or tag a one-time cold initializer `# import-time: <what>`): {c}", file=sys.stderr)
    return 1 if new else 0


def prune_baseline(ref):
    sha = _resolve(ref)
    if not sha:
        print(f"no_runtime_import_module: unknown ref {ref}", file=sys.stderr)
        return 2
    top = _git("rev-parse", "--show-toplevel").stdout.decode().strip()
    path = os.path.join(top, BASELINE)
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    current = {}
    for p, n, c in hits_at(sha):
        current.setdefault((p, c), []).append(n)
    used = Counter()
    keep, dropped = [], 0
    for ln in lines:
        m = ROW.match(ln.strip())
        if not m or ln.lstrip().startswith("#"):
            keep.append(ln)
            continue
        key = (m["path"], m["code"].strip())
        nums = current.get(key, [])
        if used[key] < len(nums):
            keep.append(format_row(key[0], nums[used[key]], key[1]))
            used[key] += 1
        else:
            dropped += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(keep) + "\n")
    print(f"no_runtime_import_module: removed {dropped} stale rows from {BASELINE}", file=sys.stderr)
    return 0


def main(argv):
    args = list(argv)
    branch = "--branch" in args
    args = [a for a in args if a != "--branch"]
    baseline_file = None
    if "--baseline" in args:
        i = args.index("--baseline")
        if i + 1 >= len(args):
            print(__doc__, file=sys.stderr)
            return 2
        baseline_file = args[i + 1]
        del args[i:i + 2]
    if args and args[0] == "--tree" and len(args) <= 2:
        return check_tree(args[1] if len(args) == 2 else "HEAD", branch, baseline_file)
    if args and args[0] == "--prune-baseline" and len(args) <= 2:
        return prune_baseline(args[1] if len(args) == 2 else "HEAD")
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
