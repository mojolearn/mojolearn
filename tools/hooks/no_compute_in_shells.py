#!/usr/bin/env python3
"""Refuse compute in shell files: a Python shell .mojo file is a passthrough.

The rule (Andrew, 2026-10-21): a SHELL file is the Python door

    bindings/**            the Python extension entry points
    **/*pyapi*.mojo        a kernel's sibling Python shell (<name>_pyapi.mojo)
    **/*_py.mojo           the same, older spelling

and may only unpack PythonObject arguments into typed values, call one typed
kernel entry and convert the typed result back. It must not loop over data or
launch work. This checker reports, in every shell file:

    loop     a line starting with `for` or `while`, unless the line itself or
             the line directly above carries `# small-loop(<what>)` (a loop over
             argument words of bounded, data-independent length) or
             `# driver-loop: over <what>` (a driver loop over model structure)
    launch   `enqueue_function`, `DeviceContext(`, `parallelize(`/`parallelize[`,
             `vectorize(`/`vectorize[`, anywhere in code (never taggable)

Comment lines and triple-quoted docstrings are skipped. Hits that were on main
2026-10-21 are listed in tools/hooks/compute_in_shells_baseline.txt as
`path:line: code`; a hit is grandfathered by a row with the same path and the
same code text (the line number is informational, so edits above a row do not
break it). The thin-* lanes move the work into kernel files and the list only
shrinks.

Modes:
    no_compute_in_shells.py --tree [REF] [--branch] [--baseline FILE]
        Scan every shell .mojo file committed at REF (default HEAD). Fail (exit 1)
        on any hit not covered by the baseline. The baseline is REF's file,
        restricted to the rows main's baseline also has, so a branch cannot grow
        it. Prints hit counts per directory. Rows no hit matches are stale.
        --branch (the pre-push hook passes it for every ref except main): judge
        only the shell files REF adds or changes over merge-base(origin/main, REF);
        a hit that was already in that file at the merge-base is main's debt.
    no_compute_in_shells.py --prune-baseline [REF]
        Rewrite the worktree baseline without its stale rows and with current line
        numbers. It never adds a row.
Exit 0 clean, 1 on findings, 2 on usage errors. Pure git and regex; no Mojo, no build.
"""
import os
import re
import subprocess
import sys
from collections import Counter

BASELINE = "tools/hooks/compute_in_shells_baseline.txt"
LOOP = re.compile(r"^\s*(for|while)\b")
LAUNCH = re.compile(r"enqueue_function|\bDeviceContext\s*\(|\b(?:sync_)?parallelize\s*[\[(]|\bvectorize\s*[\[(]")
TAG = re.compile(r"#\s*small-loop\(|#\s*driver-loop:")
ROW = re.compile(r"^(?P<path>[^:\s]+\.mojo):(?P<line>\d+):\s?(?P<code>.*)$")


def is_shell(path):
    """True for the Python door files this checker judges (see the module docstring)."""
    name = path.rsplit("/", 1)[-1]
    return path.startswith("bindings/") or "pyapi" in name or name.endswith("_py.mojo")


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
    """[(line number, kind, code)] of the untagged loops and the launches in one shell file."""
    out = []
    lines = text.splitlines()
    in_doc = None
    for n, raw in enumerate(lines, 1):
        s = raw.strip()
        if in_doc:
            if in_doc in s:
                in_doc = None
            continue
        q = '"""' if '"""' in s else ("'''" if "'''" in s else None)
        if q:
            # a line with a triple quote is docstring or string text, never a loop or a launch
            if s.count(q) % 2 == 1:
                in_doc = q
            continue
        if not s or s.startswith("#"):
            continue
        code = _code_part(raw)
        prev = lines[n - 2] if n >= 2 else ""
        if LOOP.match(code) and not (TAG.search(raw) or TAG.search(prev)):
            out.append((n, "loop", s))
        elif LAUNCH.search(code):
            out.append((n, "launch", s))
    return out


def _git(*args, inp=None):
    return subprocess.run(["git", *args], capture_output=True, input=inp)


def _resolve(ref):
    r = _git("rev-parse", "--verify", "--quiet", ref + "^{commit}")
    return r.stdout.decode().strip() if r.returncode == 0 else None


def _shell_blobs(ref):
    """{path: blob sha} of every shell .mojo file at ref."""
    r = _git("ls-tree", "-r", "--full-tree", ref)
    if r.returncode != 0:
        raise SystemExit(f"no_compute_in_shells: cannot list {ref}: {r.stderr.decode().strip()}")
    out = {}
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        meta, _, path = ln.partition("\t")
        bits = meta.split()
        if len(bits) == 3 and bits[1] == "blob" and path.endswith(".mojo") and is_shell(path):
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
    """[(path, line, kind, code)] at ref, sorted by path and line (only `paths`, if given)."""
    blobs = _shell_blobs(ref)
    if paths is not None:
        blobs = {p: s for p, s in blobs.items() if p in paths}
    texts = _read(blobs.values())
    return [(p, n, k, c) for p in sorted(blobs) for n, k, c in scan(texts[blobs[p]])]


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
        print(f"no_compute_in_shells: unknown ref {ref}", file=sys.stderr)
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
        key = (h[0], h[3])
        seen[key] += 1
        if seen[key] > allow[key]:
            new.append(h)
    stale = Counter() if branch else base - _keys(hits)
    kinds = Counter(h[2] for h in hits)
    print(f"no_compute_in_shells: {len(hits)} hits in {len({h[0] for h in hits})} shell .mojo files at {sha[:9]} "
          f"({kinds['loop']} untagged loops, {kinds['launch']} launches; baseline {sum(base.values())}); "
          f"per directory: {_per_dir(hits)}", file=sys.stderr)
    for (p, c), n in sorted(grown.items()):
        print(f"no_compute_in_shells: baseline row {p}: {c} (x{n}) is not in main's baseline; "
              f"the baseline only shrinks", file=sys.stderr)
    for (p, c), n in sorted(stale.items()):
        print(f"no_compute_in_shells: stale baseline row {p}: {c} (x{n}) matches no hit: "
              f"remove it with --prune-baseline", file=sys.stderr)
    for p, n, k, c in new:
        what = ("untagged loop in a shell file; move it into the kernel file (or tag a bounded argument-word loop "
                "`# small-loop(<what>)`, a model-structure driver loop `# driver-loop: over <what>`)"
                if k == "loop" else "kernel launch in a shell file; move it into the kernel file")
        print(f"{p}:{n}: {what}: {c}", file=sys.stderr)
    return 1 if new else 0


def prune_baseline(ref):
    sha = _resolve(ref)
    if not sha:
        print(f"no_compute_in_shells: unknown ref {ref}", file=sys.stderr)
        return 2
    top = _git("rev-parse", "--show-toplevel").stdout.decode().strip()
    path = os.path.join(top, BASELINE)
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    current = {}
    for p, n, _k, c in hits_at(sha):
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
    print(f"no_compute_in_shells: removed {dropped} stale rows from {BASELINE}", file=sys.stderr)
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
