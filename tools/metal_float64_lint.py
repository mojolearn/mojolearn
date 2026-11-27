#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""METAL FLOAT64 LINT: Float64 in Mojo code a device kernel can reach. Metal has no float64.

    python3 tools/metal_float64_lint.py [--commit <sha>] [--root DIR] [--jobs N] [--list]

WHY (lane mac-speed-1, 2026-11-27). The 0.9.0 Apple build died at the release, after every Linux leg was green:
`_mojolearn_native_sequence_m1` "Failed to verify LLVM IR for Metal" (air.convert f32->f64, s.i64<->f64), because
autograd/nn_conv_ops.mojo up_candidates computed its range in Float64 and seq_kernel reaches it through
op_nnc_up_vjp (fixed in f6a2ad489a). NVIDIA, AMD and the host take float64, so no Linux compile can see it. This
lint sees it before any compile, in seconds, on any machine.

WHAT IT FLAGS. A `Float64` / `DType.float64` / `float64` token in the code (comments, docstrings and strings
blanked) of a function reachable from a device kernel:
  * kernel roots: every name launched with `enqueue_function[<name>...]` or `compile_function[<name>...]`;
  * reachability: by NAME over every tracked .mojo outside bench/, experiments/ and docs/ (a call is an
    identifier followed by `(` or `[` that names a `def`/`fn` in the corpus). Same-named functions of different
    modules merge, so it over-approximates: a flagged site is either fixed or allow-listed with a reason;
  * guards: a line inside a `comptime if` block that cannot be compiled for Apple (`not <apple>`, or a test of
    an NVIDIA / AMD / CUDA / HIP target with no Apple term), or in the `else:` of a positive Apple test, is not
    flagged, and a call made only there is not followed.
THE ALLOW LIST (tools/metal_float64_lint_allow.txt): `<path>:<function>  # <CLASS>: <reason>`, CLASS one of
SAFE-NOT-APPLE (the function is never compiled into an Apple binding), SAFE-CONST (a Float64 compile-time constant
converted at compile time: no float64 value reaches the kernel's IR), SAFE-HOST (the Float64 is on a host-only
path the name graph cannot separate), SAFE-NOT-SHIPPED. Unlisted sites fail; a line that matches nothing is STALE
and fails. The compiled-kernel audit (packaging/macos/air_contract.py) stays the after-build proof; this is the
before-build screen, and the Apple parse of the cross-compile check (cross-compile-check.yml) is the compiler's.

PARALLEL: files are parsed across --jobs worker processes (default the host's cores; the graph walk is one
breadth-first pass over the merged index and takes milliseconds).
"""
import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOW = "tools/metal_float64_lint_allow.txt"
CLASSES = ("SAFE-NOT-APPLE", "SAFE-CONST", "SAFE-HOST", "SAFE-NOT-SHIPPED")
_SKIP_PREFIX = ("bench/", "experiments/", "docs/")
_DEF = re.compile(r"^([ \t]*)(?:def|fn)[ \t]+(\w+)")
_LAUNCH = re.compile(r"\b(?:enqueue_function|enqueue_function_checked|enqueue_function_unchecked|compile_function)"
                     r"\s*\[\s*([A-Za-z_]\w*)")
_CALL = re.compile(r"(?<![\w])([A-Za-z_]\w*)\s*[\[(]")
_F64 = re.compile(r"\bFloat64\b|\bfloat64\b")
_COMPTIME_IF = re.compile(r"^([ \t]*)(comptime[ \t]+if|elif)[ \t]+(.*):[ \t]*$")
_ELSE = re.compile(r"^([ \t]*)else[ \t]*:[ \t]*$")
_APPLE = re.compile(r"apple|APPLE|metal|METAL|Metal|Apple")
_OTHER = re.compile(r"nvidia|NVIDIA|amd|AMD|cuda|CUDA|\bhip\b|HIP|sm_\d|gfx\d")
_ALLOW_LINE = re.compile(r"^([^:#\s]+):(\w+)\s+#\s*(SAFE-[A-Z-]+):\s*\S")
#: Names too generic to follow by name (they are std methods far more often than ours).
_NOT_FOLLOWED = frozenset(("__init__", "__del__", "__copyinit__", "__moveinit__", "__getitem__", "__setitem__",
                           "load", "store", "len", "min", "max", "abs", "range", "print", "write_to", "copy"))


def in_scope(path):
    return path.endswith(".mojo") and not path.startswith(_SKIP_PREFIX)


def _blank(m):
    return "\n" * m.group(0).count("\n")


def code_only(text):
    """`text` with docstrings, comments and string literals blanked (line numbers kept)."""
    text = re.sub(r'(?s)(""".*?"""|\'\'\'.*?\'\'\')', _blank, text)
    text = re.sub(r"#[^\n]*", "", text)
    return re.sub(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', '""', text)


def _indent(line):
    return len(line) - len(line.lstrip(" \t"))


def _balance(line):
    return line.count("(") + line.count("[") - line.count(")") - line.count("]")


#: A compile-time test that the device has float64 (false on every Apple GPU).
_HAS_F64 = re.compile(r"^(not[ \t]+)?device_has_float64\(\)$")


_ALIAS = re.compile(r"^comptime[ \t]+(\w+)[ \t]*(?::[^=\n]*)?=[ \t]*([^\n]+)$", re.M)


def aliases(files):
    """{name: expression} of every module-level `comptime NAME = <expr>` that names an Apple test, so a guard
    `comptime if NAME:` is judged by its expression (isolation_forest's IDN_IF_SCORES_F64 = not TARGET_HAS_APPLE_GPU)."""
    out = {}
    for _path, text in files:
        for m in _ALIAS.finditer(text):
            expr = m.group(2).split("#")[0].strip()
            if _APPLE.search(expr) or _HAS_F64.match(expr):
                out[m.group(1)] = expr
    return out


def _expand(cond, names):
    c = cond.strip()
    neg = re.match(r"^not[ \t]+(\w+)$", c)
    if neg and neg.group(1) in names:
        inner = names[neg.group(1)]
        return inner[4:].strip() if inner.startswith("not ") else "not " + inner
    return names.get(c, c)


def cannot_be_apple(cond, names=None):
    """True when a `comptime if` condition is false on every Apple target (as far as its text shows)."""
    c = _expand(cond, names or {})
    f = _HAS_F64.match(c)
    if f:
        return not f.group(1)
    if re.match(r"^not\b", c) and _APPLE.search(c) and " or " not in c:
        return True
    return bool(_OTHER.search(c)) and not _APPLE.search(c) and " or " not in c and not c.startswith("not")


def is_apple_test(cond, names=None):
    c = _expand(cond, names or {})
    f = _HAS_F64.match(c)
    if f:
        return bool(f.group(1))
    return bool(_APPLE.search(c)) and not c.startswith("not") and " or " not in c


_FROM = re.compile(r"^[ \t]*from[ \t]+(\.*[A-Za-z_][\w.]*|\.+)[ \t]+import[ \t]+(\([^)]*\)|[^\n]+)", re.M)
_IMPORT = re.compile(r"^[ \t]*import[ \t]+([A-Za-z_][\w.]*)", re.M)
_METHOD = re.compile(r"\.([A-Za-z_]\w*)\s*[\[(]")
#: Roots a module name resolves against: the repository and bindings/ (the builds pass -I . -I bindings).
_ROOTS = ("", "bindings/")
#: Files whose kernels are not shipped (native tests, oracles): their launches are not roots.
_TEST = re.compile(r"(^|/)(native_tests|tests)/|oracle|(^|/)test_")


def imports(path, code):
    """[(module, [names] or ["*"])] of one file (code_only text)."""
    out = []
    for m in _FROM.finditer(code):
        names = [n.strip().split(" as ")[0].strip() for n in re.split(r"[,()\n]", m.group(2))]
        out.append((m.group(1), [n for n in names if n]))
    for m in _IMPORT.finditer(code):
        out.append((m.group(1), ["*"]))
    return out


def module_files(module, importer, known):
    """Tracked files MODULE can mean from IMPORTER, over KNOWN (a set of paths)."""
    if module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        base = Path(importer).parent
        for _ in range(dots - 1):
            base = base.parent
        bases = ["" if str(base) == "." else str(base) + "/"]
        parts = [p for p in module.lstrip(".").split(".") if p]
    else:
        parts = module.split(".")
        d = str(Path(importer).parent)
        bases = list(_ROOTS) + ([d + "/"] if d != "." else [])
    out = []
    for b in bases:
        stem = b + "/".join(parts)
        for cand in (stem + ".mojo", stem + "/__init__.mojo"):
            if cand in known:
                out.append(cand)
    return out


def parse(job):
    """(defs, imports, launches) of one file. defs: [(name, first line, [f64 lines], {calls}, {method calls})] for
    every def/fn with the lines of its own body an Apple compile can reach (a nested def is its own entry)."""
    path, text, names = job
    code = code_only(text)
    lines = code.split("\n")
    defs = []
    launches = set()
    stack = []   # [(def indent, entry)]
    guard = []   # [(indent of the `comptime if`, skip body, skip else)]
    pending_else = {}
    depth = 0    # open brackets of a def signature that spans lines (its `):` may sit at the def's indent)
    for ln, line in enumerate(lines):
        if not line.strip():
            continue
        if depth > 0:
            depth += _balance(line)
            if stack and _F64.search(line) and not any(g[1] for g in guard):
                stack[-1][1][2].append(ln + 1)   # a Float64 argument or result type
            continue
        ind = _indent(line)
        while stack and ind <= stack[-1][0]:
            stack.pop()
        while guard and ind <= guard[-1][0]:
            g = guard.pop()
            pending_else[g[0]] = g
        m = _COMPTIME_IF.match(line)
        e = _ELSE.match(line)
        if m and (m.group(2) != "elif" or ind in pending_else):
            cond = m.group(3)
            guard.append((ind, cannot_be_apple(cond, names), is_apple_test(cond, names)))
            pending_else.pop(ind, None)
            continue
        if e and ind in pending_else:
            g = pending_else.pop(ind)
            guard.append((ind, g[2], False))
            continue
        pending_else = {k: v for k, v in pending_else.items() if k < ind}
        d = _DEF.match(line)
        if d:
            entry = (d.group(2), ln + 1, [], set(), set())
            defs.append(entry)
            stack.append((ind, entry))
            depth = max(0, _balance(line))
            if _F64.search(line) and not any(g[1] for g in guard):
                entry[2].append(ln + 1)
            continue
        skipped = any(g[1] for g in guard)
        if not skipped:
            for k in _LAUNCH.finditer(line):
                launches.add(k.group(1))
        if not stack or skipped:
            continue
        entry = stack[-1][1]
        if _F64.search(line):
            entry[2].append(ln + 1)
        for c in _CALL.finditer(line):
            entry[3].add(c.group(1))
        for c in _METHOD.finditer(line):
            entry[4].add(c.group(1))
    return path, defs, imports(path, code), (set() if _TEST.search(path) else launches)


def scan(files, jobs=None):
    """([(path, line, function)] of every Float64 a shipped kernel can reach, roots) over FILES [(path, text)].

    A call `f(` from file F resolves to F's own `f` and to `f` in every file F imports it from (`from m import f`,
    `from m import *`, `import m`); a method call `.f(` to `f` in F and in every file F imports from."""
    workers = max(1, jobs or os.cpu_count() or 1)
    names = aliases(files)
    files = [(p, t, names) for p, t in files]
    if workers == 1 or len(files) < 64:
        parsed = list(map(parse, files))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            parsed = list(ex.map(parse, files, chunksize=32))
    known = {p for p, _, _, _ in parsed}
    by_file = {}      # path -> {name: [entry]}
    links = {}        # path -> {name: [files]} for imported names, and "*" -> files of star/module imports
    for path, defs, imps, _l in parsed:
        table = by_file.setdefault(path, {})
        for entry in defs:
            table.setdefault(entry[0], []).append(entry)
        lk = links.setdefault(path, {})
        for module, names in imps:
            targets = module_files(module, path, known)
            for n in names:
                lk.setdefault(n, []).extend(targets)
    def callees(path, name, method):
        out = [(path, e) for e in by_file[path].get(name, [])]
        lk = links[path]
        sources = set(lk.get(name, [])) | set(lk.get("*", []))
        if method:
            sources |= {f for fs in lk.values() for f in fs}
        for f in sources:
            out.extend((f, e) for e in by_file.get(f, {}).get(name, []))
        return out
    roots, todo = 0, []
    for path, _d, _i, launches in parsed:
        for name in launches:
            hits = callees(path, name, False)
            roots += 1 if hits else 0
            todo.extend(hits)
    seen, found = set(), set()
    while todo:
        path, entry = todo.pop()
        if (path, entry[0], entry[1]) in seen:
            continue
        seen.add((path, entry[0], entry[1]))
        found.update((path, ln, entry[0]) for ln in entry[2])
        for name in entry[3]:
            if name not in _NOT_FOLLOWED:
                todo.extend(callees(path, name, False))
        for name in entry[4]:
            if name not in _NOT_FOLLOWED:
                todo.extend(callees(path, name, True))
    return sorted(found), roots


def read_allow(text):
    allow, bad = {}, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ALLOW_LINE.match(line)
        if not m or m.group(3) not in CLASSES:
            bad.append(raw)
            continue
        allow[(m.group(1), m.group(2))] = raw
    return allow, bad


def judge(found, allow):
    keys = {(p, f) for p, _, f in found}
    return [s for s in found if (s[0], s[2]) not in allow], sorted(set(allow) - keys)


def load(root, commit):
    if commit:
        names = [n for n in subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=root,
                                                    text=True).split("\n") if in_scope(n)]
        batch = subprocess.run(["git", "cat-file", "--batch"], cwd=root, check=True, capture_output=True,
                               input="".join(f"{commit}:{n}\n" for n in names).encode()).stdout
        texts, pos = [], 0
        for _ in names:
            header_end = batch.index(b"\n", pos)
            size = int(batch[pos:header_end].split()[2])
            texts.append(batch[header_end + 1:header_end + 1 + size].decode("utf8", "replace"))
            pos = header_end + 1 + size + 1
        allow = subprocess.run(["git", "show", f"{commit}:{ALLOW}"], cwd=root, capture_output=True).stdout.decode()
        return list(zip(names, texts)), allow
    names = [n for n in subprocess.check_output(["git", "ls-files", "*.mojo"], cwd=root, text=True).split("\n")
             if in_scope(n)]
    allow_path = Path(root) / ALLOW
    return ([(n, (Path(root) / n).read_text(errors="replace")) for n in names],
            allow_path.read_text() if allow_path.exists() else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default="")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--jobs", type=int, default=0, help="parser processes (default: the host's cores)")
    ap.add_argument("--list", action="store_true", help="print every reachable Float64 site")
    a = ap.parse_args(argv)
    files, allow_text = load(a.root, a.commit)
    found, roots = scan(files, a.jobs or None)
    allow, bad = read_allow(allow_text)
    unlisted, stale = judge(found, allow)
    if a.list:
        for p, ln, fn in found:
            print(f"F64LINT SITE {p}:{ln} {fn}")
    for raw in bad:
        print(f"F64LINT FAIL malformed allow line: {raw}")
    for p, ln, fn in unlisted:
        print(f"F64LINT FAIL {p}:{ln} {fn}: Float64 in code a device kernel reaches (Metal has no float64) -- use"
              f" Float32 or an integer form, guard it off Apple, or allow-list it with a reason in {ALLOW}")
    for p, fn in stale:
        print(f"F64LINT FAIL stale allow line (matches nothing): {p}:{fn}")
    ok = not (unlisted or stale or bad)
    print(f"F64LINT {'PASS' if ok else 'FAIL'} {len(files)} files, {roots} kernel roots, {len(found)} reachable"
          f" Float64 sites, {len(found) - len(unlisted)} allowed, {len(unlisted)} unlisted, {len(stale)} stale")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
