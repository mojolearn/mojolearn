#!/usr/bin/env python3
"""Refuse Python in kernel code: a kernel .mojo file must not import Python.

The test (Andrew, 2026-10-21): a `.mojo` file must not contain a line matching
`^\\s*from (std\\.)?python`, `^\\s*import (std\\.)?python`, or `external_call["Py`,
unless it is a shell file. Shell files are the Python door and may import Python:

    bindings/**            the Python extension entry points
    **/*pyapi*.mojo        a kernel's sibling Python shell (<name>_pyapi.mojo)
    **/*_py.mojo           the same, older spelling
    bench/**, experiments/**, **/checks/**
                           tooling, not product

Every other `.mojo` file is kernel code. Files that failed on main 9f9bdb4a8 are
listed in tools/hooks/python_in_kernels_baseline.txt; they may stay until they are
purged, and the list only shrinks.

Modes:
    no_python_in_kernels.py --tree [REF] [--branch] [--baseline FILE]
        Scan every .mojo file committed at REF (default HEAD). Fail (exit 1) on any
        hit not in the baseline. The baseline is REF's file, restricted to the rows
        main's baseline also has, so a branch cannot grow it. Prints the hit count
        per directory. Rows whose file no longer matches are reported as stale.
        --branch (the pre-push hook passes it for every ref except main): judge
        only the .mojo files REF adds or changes over merge-base(origin/main, REF);
        a file that already matched at the merge-base is main's, not the branch's.
    no_python_in_kernels.py --prune-baseline [REF]
        Rewrite the worktree baseline without its stale rows. It never adds a row.
Exit 0 clean, 1 on findings, 2 on usage errors. Pure git and regex; no Mojo, no build.
"""
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = "tools/hooks/python_in_kernels_baseline.txt"
PATTERN = re.compile(r'^\s*from (std\.)?python|^\s*import (std\.)?python|external_call\["Py', re.M)
SHELL_PREFIXES = ("bindings/", "bench/", "experiments/")


def is_shell(path):
    """True for the files allowed to import Python (see the module docstring)."""
    name = path.rsplit("/", 1)[-1]
    if path.startswith(SHELL_PREFIXES):
        return True
    if "pyapi" in name or name.endswith("_py.mojo"):
        return True
    return path.startswith("checks/") or "/checks/" in path


def _git(*args, inp=None):
    return subprocess.run(["git", *args], capture_output=True, input=inp)


def _resolve(ref):
    r = _git("rev-parse", "--verify", "--quiet", ref + "^{commit}")
    return r.stdout.decode().strip() if r.returncode == 0 else None


def _mojo_blobs(ref):
    """{path: blob sha} of every kernel (non-shell) .mojo file at ref."""
    r = _git("ls-tree", "-r", "--full-tree", ref)
    if r.returncode != 0:
        raise SystemExit(f"no_python_in_kernels: cannot list {ref}: {r.stderr.decode().strip()}")
    out = {}
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        meta, _, path = ln.partition("\t")
        bits = meta.split()
        if len(bits) == 3 and bits[1] == "blob" and path.endswith(".mojo") and not is_shell(path):
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
    """Sorted kernel .mojo paths at ref that import or call Python (only `paths`, if given)."""
    blobs = _mojo_blobs(ref)
    if paths is not None:
        blobs = {p: s for p, s in blobs.items() if p in paths}
    texts = _read(blobs.values())
    return sorted(p for p, s in blobs.items() if PATTERN.search(texts[s]))


def parse_baseline(text):
    return {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")}


def _baseline_at(ref):
    r = _git("show", f"{ref}:{BASELINE}")
    return parse_baseline(r.stdout.decode("utf-8", "replace")) if r.returncode == 0 else None


def _main_ref():
    for m in ("refs/remotes/origin/main", "origin/main", "refs/heads/main"):
        if _resolve(m):
            return m
    return None


def _per_dir(paths):
    c = Counter(p.split("/", 1)[0] if "/" in p else "." for p in paths)
    return ", ".join(f"{d} {n}" for d, n in sorted(c.items())) or "none"


def check_tree(ref, branch=False, baseline_file=None):
    sha = _resolve(ref)
    if not sha:
        print(f"no_python_in_kernels: unknown ref {ref}", file=sys.stderr)
        return 2
    if baseline_file:
        with open(baseline_file, encoding="utf-8") as f:
            base = parse_baseline(f.read())
    else:
        base = _baseline_at(sha) or set()
    main = _main_ref()
    main_base = _baseline_at(main) if main else None
    if main_base is not None:
        grown = sorted(base - main_base)
        base &= main_base  # the baseline only shrinks: rows main lacks are not grandfathered
    else:
        grown = []
    hits = hits_at(sha)
    judged, old = set(hits), set()
    if branch and main:
        mb = _git("merge-base", main, sha).stdout.decode().strip()
        if mb and mb != sha:
            changed = _git("diff", "--name-only", "--no-renames", mb, sha, "--", "*.mojo").stdout.decode().split()
            judged &= set(changed)
            old = set(hits_at(mb, judged))
    new = sorted(judged - base - old)
    stale = sorted(base - set(hits)) if not branch else []
    print(f"no_python_in_kernels: {len(hits)} kernel .mojo files import Python at {sha[:9]} "
          f"(baseline {len(base)}); per directory: {_per_dir(hits)}", file=sys.stderr)
    for p in grown:
        print(f"no_python_in_kernels: baseline row {p} is not in main's baseline; the baseline only shrinks",
              file=sys.stderr)
    for p in stale:
        print(f"no_python_in_kernels: stale baseline row {p} (no longer imports Python): "
              f"remove it with --prune-baseline", file=sys.stderr)
    for p in new:
        print(f"{p}: kernel .mojo file imports or calls Python; move the Python door to bindings/ or a "
              f"sibling <name>_pyapi.mojo shell", file=sys.stderr)
    return 1 if new else 0


def prune_baseline(ref):
    sha = _resolve(ref)
    if not sha:
        print(f"no_python_in_kernels: unknown ref {ref}", file=sys.stderr)
        return 2
    top = _git("rev-parse", "--show-toplevel").stdout.decode().strip()
    path = os.path.join(top, BASELINE)
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    hits = set(hits_at(sha))
    keep = [ln for ln in lines if not ln.strip() or ln.lstrip().startswith("#") or ln.strip() in hits]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(keep) + "\n")
    print(f"no_python_in_kernels: removed {len(lines) - len(keep)} stale rows from {BASELINE}", file=sys.stderr)
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
