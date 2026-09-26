#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""No call cycle may reach a MAX library GPU kernel (linalg, nn).

    python3 tools/check_no_recursion_around_kernels.py            # whole repo
    python3 tools/check_no_recursion_around_kernels.py --all      # + cycles reaching our kernels only
    python3 tools/check_no_recursion_around_kernels.py FILE...    # these files only

THE HANG (Mojo 1.0.0, 0.8.19 FAST). `svm/impl/distance/kernel_matrices.mojo`
had `kernel_op -> _kernel_rows -> kernel_op`, and FAST `kernel_op` reached
`linalg.matmul.matmul[target="gpu"]`. `mojo build --target-accelerator sm_89`
(and gfx942) spent about 30 s of CPU and then waited on a semaphore forever.
The 57-line reproducer and the report draft are in
`~/mojolearn-evidence/modular-bug-reports/`. The same cycle around our own
hand-written GEMM builds, and so does the cycle when the recursion is split by
a comptime parameter (`op[True] -> _rows -> op[False]`), because then no
function instantiation is its own ancestor. Direct self recursion
(`op -> op`) and a loop also build. Marking `op` @no_inline or
@always_inline, moving `matmul` into a helper (same file or another module),
instantiating the same `matmul` first from a non-recursive path, `-j 2`,
`--emit object|llvm`, a plain executable and a Linux host triple all still
hang. Mojo 1.1.0 hangs too; the 1.2.0.dev2026092505 nightly builds it for
sm_89 and gfx942 in about 30 s. It is a compiler bug, not a documented
limit, and this check is the guard until the fix reaches a stable release.

NOT EVERY FLAGGED CYCLE HANGS. The k-means cycle below reaches `matmul` too
and compiles; what separates it from the reproducer is not known. A flagged
cycle is therefore reviewed with a measured cross compile (or broken), not
assumed to hang.

THE RULE. A function that is its own caller (directly or through helpers in
the repo) must not reach a name imported from `linalg` or `nn`. Split the
recursion by a comptime parameter, make it a loop, or gate the recursive call
out of the build mode that reaches the library kernel, and list the reviewed
cycle in REVIEWED below with the reason.

THE ANALYSIS. Textual and conservative: every `def` (module level, struct
method or nested) is a node keyed by (file, name); a call edge is `name(` or
`name[` where `name` is defined in the same file or imported by
`from x.y import name` from a file in this repo. Comptime branches are not
evaluated, which is why a reviewed cycle is listed rather than inferred.

Exit 0 when every offending cycle is reviewed, 1 when one is not, 2 when no
Mojo file was read (a check that read nothing did not pass).
"""

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modules whose GPU entry points we treat as library kernels. `linalg.matmul`
# is the reproduced one; the rest share its comptime dispatch machinery.
LIBRARY_KERNEL_PREFIXES = ("linalg", "nn")

# (file, sorted cycle members) -> reason. A reviewed cycle must say why the
# recursive edge cannot coexist with the library kernel in one instantiation.
REVIEWED = {
    ("svm/impl/distance/kernel_matrices.mojo", ("_kernel_rows", "kernel_op")): (
        "the recursive call is `comptime if GLOBAL_NUMERIC_MODE == "
        "NUMERIC_IDENTICAL`, and IDENTICAL reaches identical_gemm_into, "
        "never gemm_nt/matmul (FAST)"
    ),
    ("cluster/impl/detail/kmeans.mojo",
     ("init_scalable_kmeans_plus_plus", "kmeans_fit_main_traced")): (
        "runtime-gated only (the inner fit is INIT_ARRAY), and FAST reaches "
        "matmul through core.gemm.gemm_nt, but MEASURED to compile: FAST "
        "_mojolearn for sm_89 in 101 s on Mojo 1.0.0 (2026-09-25). Re-measure "
        "with tools/cross_compile_check.py if either function changes shape"
    ),
}

DEF_RE = re.compile(r"^(\s*)def\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\[(]")
FROM_RE = re.compile(r"^\s*from\s+([.A-Za-z_][.A-Za-z0-9_]*)\s+import\s+(.*)$")
CALL_RE = re.compile(r"(?<![.\w])([A-Za-z_][A-Za-z0-9_]*)\s*[\[(]")
KERNEL_LAUNCH_RE = re.compile(r"\benqueue_function\w*\b|\bcompile_function\w*\b")


def _strip_comment(line):
    """The line without its comment and with string contents blanked, so
    `print("check_x(...)")` is not a call of check_x."""
    out, quote, escaped = [], None, False
    for ch in line:
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
                out.append(ch)
                continue
            out.append(" ")
            continue
        if ch in "\"'":
            quote = ch
        elif ch == "#":
            break
        out.append(ch)
    return "".join(out)


def _code_lines(text):
    """Source lines with comments and docstrings blanked (line count kept)."""
    lines = text.split("\n")
    out = []
    in_doc = None
    for line in lines:
        if in_doc:
            if in_doc in line:
                in_doc = None
            out.append("")
            continue
        s = line.strip()
        for q in ('"""', "'''"):
            if s.startswith(q) or s.startswith("r" + q):
                body = s[s.index(q) + 3:]
                if q not in body:
                    in_doc = q
                out.append("")
                break
        else:
            out.append(_strip_comment(line))
    return out


def _imports(lines, relpath):
    """local name -> (module path, original name) for `from X import a, b`."""
    result = {}
    i = 0
    while i < len(lines):
        m = FROM_RE.match(lines[i])
        if not m:
            i += 1
            continue
        module, names = m.group(1), m.group(2)
        if "(" in names and ")" not in names:
            j = i + 1
            while j < len(lines) and ")" not in lines[j]:
                names += " " + lines[j]
                j += 1
            if j < len(lines):
                names += " " + lines[j]
            i = j
        names = names.replace("(", " ").replace(")", " ")
        for part in names.split(","):
            part = part.strip()
            if not part or part == "*":
                continue
            bits = part.split()
            orig = bits[0]
            local = bits[2] if len(bits) >= 3 and bits[1] == "as" else orig
            result[local] = (module, orig)
        i += 1
    return result


def _resolve_module(module, relpath, known):
    """Repo-relative file for a module path, or None if outside the repo."""
    if module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        base = os.path.dirname(relpath)
        for _ in range(dots - 1):
            base = os.path.dirname(base)
        rest = module.lstrip(".")
        parts = [base] + (rest.split(".") if rest else [])
    else:
        parts = module.split(".")
    stem = os.path.normpath(os.path.join(*parts)) if parts else ""
    for cand in (stem + ".mojo", os.path.join(stem, "__init__.mojo")):
        if cand in known:
            return cand
    return None


def parse(sources):
    """sources: {relpath: text}. Returns (nodes, edges, lib_calls, launches).

    nodes: set of (file, name); edges: node -> set(node); lib_calls: node ->
    set of 'module.name' library kernels called directly; launches: nodes that
    launch a kernel of ours (enqueue_function / compile_function)."""
    known = set(sources)
    defs = {}
    bodies = {}
    imports = {}
    for rel, text in sources.items():
        lines = _code_lines(text)
        imports[rel] = _imports(lines, rel)
        stack = []  # (indent, name, start)
        spans = []
        for n, line in enumerate(lines):
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            while stack and indent <= stack[-1][0] and not line.lstrip().startswith((")", "]")):
                ind, name, start = stack.pop()
                spans.append((name, start, n))
            m = DEF_RE.match(line)
            if m:
                # A nested def (closure) stays inside its parent's span and
                # is not a node: its calls are made on the parent's behalf.
                stack.append((len(m.group(1)), None if stack else m.group(2), n))
        while stack:
            ind, name, start = stack.pop()
            spans.append((name, start, len(lines)))
        names = set()
        for name, start, end in spans:
            if name is None:
                continue
            names.add(name)
            bodies.setdefault((rel, name), []).append("\n".join(lines[start:end]))
        defs[rel] = names

    edges, lib_calls, launches = {}, {}, set()
    for (rel, name), chunks in bodies.items():
        key = (rel, name)
        edges.setdefault(key, set())
        body = "\n".join(chunks)
        # Skip the def line's own name so `def f(` is not a self call.
        body_calls = CALL_RE.findall(re.sub(r"\bdef\s+\w+", "", body))
        for callee in set(body_calls):
            if callee in defs[rel]:
                edges[key].add((rel, callee))
            elif callee in imports[rel]:
                module, orig = imports[rel][callee]
                if module.split(".")[0] in LIBRARY_KERNEL_PREFIXES:
                    lib_calls.setdefault(key, set()).add(module + "." + orig)
                    continue
                target = _resolve_module(module, rel, known)
                if target and orig in defs.get(target, ()):
                    edges[key].add((target, orig))
        if KERNEL_LAUNCH_RE.search(body):
            launches.add(key)
    return set(bodies), edges, lib_calls, launches


def _sccs(nodes, edges):
    index, low, on, stack, out = {}, {}, set(), [], []
    counter = [0]
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))

    def visit(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in edges.get(v, ()):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for v in sorted(nodes):
        if v not in index:
            visit(v)
    return out


def _reach(start, edges):
    """Every node reachable from the nodes in `start`, including them."""
    seen = set(start)
    frontier = list(start)
    while frontier:
        v = frontier.pop()
        for w in edges.get(v, ()):
            if w not in seen:
                seen.add(w)
                frontier.append(w)
    return seen


def find_cycles(sources):
    """[(members, library kernels reached, reaches our launches)] for every
    call cycle in `sources`."""
    nodes, edges, lib_calls, launches = parse(sources)
    found = []
    for comp in _sccs(nodes, edges):
        if len(comp) == 1 and comp[0] not in edges.get(comp[0], ()):
            continue
        reach = _reach(comp, edges)
        libs = sorted({lib for v in reach for lib in lib_calls.get(v, ())})
        ours = any(v in launches for v in reach)
        found.append((sorted(comp), libs, ours))
    return found


def _reviewed(members):
    files = {f for f, _ in members}
    if len(files) != 1:
        return None
    key = (files.pop(), tuple(sorted(n for _, n in members)))
    return REVIEWED.get(key)


def _repo_sources(paths):
    if paths:
        rels = [os.path.relpath(os.path.abspath(p), ROOT) for p in paths]
    else:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "*.mojo"],
                             capture_output=True, text=True, check=True).stdout
        rels = [line for line in out.split("\n") if line]
    sources = {}
    for rel in rels:
        try:
            with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
                sources[os.path.normpath(rel)] = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
    return sources


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--all", action="store_true",
                    help="also list cycles that reach only our own kernels")
    args = ap.parse_args(argv)
    sources = _repo_sources(args.paths)
    if not sources:
        print("check_no_recursion_around_kernels: read no Mojo file", file=sys.stderr)
        return 2
    bad = 0
    for members, libs, ours in find_cycles(sources):
        label = " -> ".join(f"{f}:{n}" for f, n in members)
        if libs:
            why = _reviewed(members)
            if why:
                print(f"REVIEWED  {label}\n          reaches {', '.join(libs)}; {why}")
            else:
                bad += 1
                print(f"FAIL      {label}\n          reaches {', '.join(libs)}")
        elif args.all and ours:
            print(f"INFO      {label}\n          reaches our own kernel launches only")
    print(f"check_no_recursion_around_kernels: {len(sources)} files, "
          f"{bad} unreviewed cycle(s) reaching linalg/nn kernels")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
