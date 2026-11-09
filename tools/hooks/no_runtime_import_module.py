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

Since 2026-11-09 (lane p4-python-enforcement) three more runtime module reads
are judged the same way, because they bypassed the rule (audit
py-audit-a-imports section 5):

    loaded_module   `native_loaded_module(<name>)` of a module that is not a
                    C-implemented module (C_MODULES below): a Python-coded
                    module such as `mojolearn._mode`, `os` or `numpy`, or a
                    name that is not a string literal
    sys_modules     a `sys.modules` read on a call path
                    (`native_loaded_module("sys").modules`, `.module("sys").modules`)
    module_fields   `py_fields(<module>)`: a module's dict read on a call path
                    (`py_fields(native_loaded_module(...))`, `py_fields(module)`)

Each hit has one kind (the first that matches in KINDS order). The baseline
header line `# kinds: ...` names the kinds it covers. A kind the baseline of
main does not list yet is bootstrapped from the pushed tree's rows once; from
then on every kind only shrinks.

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
    no_runtime_import_module.py --write-baseline [REF]
        Rewrite the worktree baseline as every hit at REF with the `# kinds:`
        header (to bootstrap a new kind, or re-take it at a merge); the check
        still refuses rows main lacks in a kind main covers.
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
# Modules implemented in C (or holding only C objects): reading them at run time
# executes no Python code. Everything else reached by native_loaded_module is a
# Python-coded module read.
C_MODULES = frozenset({
    "builtins", "operator", "_operator", "sys", "array", "itertools", "math", "posix", "_thread",
    "mmap", "_struct", "_codecs", "_json", "_io", "gc", "time", "errno", "_functools", "_collections",
    "zlib", "binascii", "_random", "_hashlib", "_sha256", "_bisect", "_heapq", "cmath", "marshal",
    "_weakref", "atexit", "_signal", "select", "fcntl", "_locale", "_string", "_abc", "_warnings",
})
LOADED = re.compile(r"\bnative_loaded_module\s*\(\s*(?P<arg>\"[^\"]*\"|'[^']*'|[^)]*)\)")
SYS_MODULES = re.compile(r"""(?:native_loaded_module|\.module)\s*\(\s*["']sys["']\s*\)\s*\.\s*modules\b""")
MODULE_FIELDS = re.compile(r"\bpy_fields\s*\(\s*(?:native_loaded_module\s*\(|Python\s*\.\s*import_module\s*\(|"
                           r"(?:module|mod|\w+_module|\w+_mod)\s*\))")
KINDS = ("import_module", "loaded_module", "sys_modules", "module_fields")
KINDS_HEADER = re.compile(r"^#\s*kinds:\s*(?P<kinds>.*)$")


def kind_of(code):
    """The KINDS entry a code line (comment stripped) hits, or None."""
    if CALL.search(code):
        return "import_module"
    for m in LOADED.finditer(code):
        arg = m["arg"].strip()
        if not (arg[:1] in "\"'" and len(arg) >= 2 and arg[1:-1] in C_MODULES):
            return "loaded_module"
    if SYS_MODULES.search(code):
        return "sys_modules"
    if MODULE_FIELDS.search(code):
        return "module_fields"
    return None
TAG = re.compile(r"#\s*import-time:")
DEF = re.compile(r"^(?P<indent>\s*)(?:def|fn)\s+(?P<name>\w+)")
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


# The file that implements native_loaded_module itself (its dynamic lookups are the mechanism).
MECHANISM_FILES = frozenset({"bindings/native_loaded_module.mojo"})


def scan(text, path=""):
    """[(line number, code)] of the runtime module reads (KINDS) in one file."""
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
        kind = None if m else kind_of(code)  # a def line declares, never reads
        if kind is None or (kind == "loaded_module" and path in MECHANISM_FILES):
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
    return [(p, n, c) for p in sorted(blobs) for n, c in scan(texts[blobs[p]], p)]


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


def parse_kinds(text):
    """The kinds a baseline text covers (`# kinds:` header; before 2026-11-09 only import_module)."""
    for ln in text.splitlines():
        m = KINDS_HEADER.match(ln.strip())
        if m:
            return frozenset(m["kinds"].replace(",", " ").split())
    return frozenset({"import_module"})


def _row_kind(row):
    return kind_of(row[-1]) or "import_module"


def _baseline_text_at(ref):
    r = _git("show", f"{ref}:{BASELINE}")
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def _baseline_at(ref):
    text = _baseline_text_at(ref)
    return parse_baseline(text) if text is not None else None


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
    main_text = _baseline_text_at(main) if main else None
    grown = Counter()
    if main_text is not None:
        main_base = _keys(parse_baseline(main_text))
        main_kinds = parse_kinds(main_text)
        # a kind main's baseline does not cover yet is bootstrapped from this tree's rows, once
        boot = Counter({k: n for k, n in base.items() if (kind_of(k[1]) or "import_module") not in main_kinds})
        grown = (base - boot) - main_base
        base = (base & main_base) + boot  # otherwise the baseline only shrinks
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
    per_kind = Counter(kind_of(h[2]) for h in hits)
    kinds = ", ".join(f"{k} {per_kind[k]}" for k in KINDS)
    print(f"no_runtime_import_module: {len(hits)} runtime module reads ({kinds}) in {len({h[0] for h in hits})} of "
          f"{len(_blobs(sha))} bindings/providers .mojo files scanned at {sha[:9]} (baseline {sum(base.values())}); "
          f"per directory: {_per_dir(hits)}", file=sys.stderr)
    for (p, c), n in sorted(grown.items()):
        print(f"no_runtime_import_module: baseline row {p}: {c} (x{n}) is not in main's baseline; "
              f"the baseline only shrinks", file=sys.stderr)
    for (p, c), n in sorted(stale.items()):
        print(f"no_runtime_import_module: stale baseline row {p}: {c} (x{n}) matches no hit: "
              f"remove it with --prune-baseline", file=sys.stderr)
    for p, n, c in new:
        print(f"{p}:{n}: runtime module read ({kind_of(c)}); reach C modules through native_loaded_module(...) and "
              f"hand Python-coded module state to the native owner at import (or tag a one-time cold initializer "
              f"`# import-time: <what>`): {c}", file=sys.stderr)
    return 1 if new or grown else 0


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


def write_baseline(ref):
    """Rewrite the worktree baseline as every current hit, with the `# kinds:` header.
    The check still refuses any row main's baseline lacks in a kind main covers, so
    this can bootstrap a new kind or re-take line numbers, never grandfather new debt."""
    sha = _resolve(ref)
    if not sha:
        print(f"no_runtime_import_module: unknown ref {ref}", file=sys.stderr)
        return 2
    top = _git("rev-parse", "--show-toplevel").stdout.decode().strip()
    path = os.path.join(top, BASELINE)
    head = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            head = [ln for ln in f.read().splitlines()
                    if ln.lstrip().startswith("#") and not KINDS_HEADER.match(ln.strip())]
    hits = hits_at(sha)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head + ["# kinds: " + " ".join(KINDS)] + [format_row(*h) for h in hits]) + "\n")
    print(f"no_runtime_import_module: wrote {len(hits)} rows to {BASELINE}", file=sys.stderr)
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
    if args and args[0] == "--write-baseline" and len(args) <= 2:
        return write_baseline(args[1] if len(args) == 2 else "HEAD")
    if args and args[0] == "--prune-baseline" and len(args) <= 2:
        return prune_baseline(args[1] if len(args) == 2 else "HEAD")
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
