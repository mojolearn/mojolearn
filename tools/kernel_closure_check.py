#!/usr/bin/env python3
"""Compile proof that kernel code builds with no Python (Andrew, 2026-10-21).

"Strip the python import and see if it compiles." This tool builds one Mojo
program that imports every kernel module, in a copy of the source tree where

  1. every SHELL file (bindings/**, **/*pyapi*.mojo, **/*_py.mojo) that imports
     or calls Python, directly or through its imports, is deleted, so a kernel
     module that reaches such a shell fails to resolve (a Python-free helper
     that lives under bindings/, such as core/hostptr.mojo, stays and is
     listed as a helper to move out of the shell); and
  2. every `from (std.)python ...` / `import (std.)python ...` line is removed from
     every remaining .mojo file (an indented one becomes `pass`, so blocks stay
     valid), so a kernel module that uses PythonObject, Python.import_module,
     GILReleased or any other Python name fails to compile.

A kernel module is a .mojo file outside the shell allowlist and outside bench/,
experiments/ and **/checks/ (tooling), that does not define `main`, and whose
path segments are identifiers (directories need no __init__.mojo: x_prep/ has
none and the bindings import it). Kernel files that are not importable this way are counted and
listed in the summary, never silently dropped.

Before the compile, a static pass reports, from the source alone:
  * kernel files that import Python or call `external_call["Py...` (the
    no-Python-in-kernels debt; an external_call cannot be stripped), and
  * kernel files that import a shell or tooling file that reaches Python (fail),
  * kernel files that import a Python-free shell file (advisory).
After the compile, the binary is inspected like tools/native_only_smoke.py
does: no undefined CPython symbol (`nm -u`) and no libpython dependency
(`ldd` / `otool -L`); then it runs once.

The generated program is written to checks/kernel_closure_main.mojo in the
checkout (git-ignored) and into the stripped tree. Mojo builds for the CPU
target (no --target-accelerator), with -I on the stripped tree.

Usage:
    python3 tools/kernel_closure_check.py [--ref REF] [--out DIR] [--mojo MOJO]
                                          [--timeout SECONDS] [--emit-only]
    --ref       commit whose files are checked (default HEAD; files come from git,
                not from the worktree, so a sparse or dirty worktree is fine)
    --out       results directory (default ./kernel-closure-results): tree/ (the
                stripped source), kernel_closure_main.mojo, build.log, symbols.log,
                dependencies.log, run.log, summary.json
    --emit-only write the program and the stripped tree, print the program path,
                and stop before the compile (the build command goes to stderr)
    --timeout   compile budget in seconds (default 2700, the 45-minute budget)

The last stderr line is one summary: `KERNEL_CLOSURE status=pass|fail ...`.
Exit 0 pass (or emitted), 1 fail, 2 usage or setup error.
"""
import argparse
import json
import keyword
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY_LINE = re.compile(r"^\s*from (std\.)?python|^\s*import (std\.)?python")
PY_LINE_M = re.compile(PY_LINE.pattern, re.M)
PY_CALL = re.compile(r'external_call\["Py')
MAIN_DEF = re.compile(r"^(?:def|fn)\s+main\s*\(", re.M)
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PYTHON_LIB = re.compile(r"(?:libpython|Python\.framework|python[23](?:\.\d+)?(?:$|/))", re.I)
PYTHON_SYMBOL = re.compile(r"\b_?(?:Py[A-Z_][A-Za-z0-9_]*|_Py[A-Za-z0-9_]+)\b")
FROM_IMPORT = re.compile(r"^\s*from\s+(\.*[A-Za-z0-9_.]*)\s+import\s+(.*)$")
PLAIN_IMPORT = re.compile(r"^\s*import\s+(.*)$")
PROGRAM_REL = "checks/kernel_closure_main.mojo"
GENERATED = {"tokenizer/impl/unicode_table_generated.mojo": ["sh", "tokenizer/tools/gen_unicode_table.sh"]}
FLAGS = ["-D", "MOJOLEARN_NUMERIC_IDENTICAL=1"]


def is_shell(path):
    name = path.rsplit("/", 1)[-1]
    return path.startswith("bindings/") or "pyapi" in name or name.endswith("_py.mojo")


def is_tooling(path):
    return path.startswith(("bench/", "experiments/", "checks/")) or "/checks/" in path


def git(*args, inp=None, cwd=None):
    return subprocess.run(["git", *args], capture_output=True, input=inp, cwd=cwd)


def read_tree(ref, cwd):
    """{path: text} of every .mojo file committed at ref."""
    r = git("ls-tree", "-r", "--full-tree", ref, cwd=cwd)
    if r.returncode:
        raise SystemExit(f"kernel_closure_check: cannot list {ref}: {r.stderr.decode().strip()}")
    blobs = {}
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        meta, _, path = ln.partition("\t")
        bits = meta.split()
        if len(bits) == 3 and bits[1] == "blob" and path.endswith(".mojo"):
            blobs[path] = bits[2]
    need = list(dict.fromkeys(blobs.values()))
    out = git("cat-file", "--batch", inp="".join(s + "\n" for s in need).encode(), cwd=cwd).stdout
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
    return {p: texts[s] for p, s in blobs.items()}


def strip_python(text):
    """(text with every Python import statement removed, number of statements removed)."""
    lines = text.split("\n")
    removed, i = 0, 0
    while i < len(lines):
        if not PY_LINE.match(lines[i]):
            i += 1
            continue
        removed += 1
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        j = i
        depth = lines[i].count("(") - lines[i].count(")")
        while (depth > 0 or lines[j].rstrip().endswith("\\")) and j + 1 < len(lines):
            j += 1
            depth += lines[j].count("(") - lines[j].count(")")
        for k in range(i, j + 1):
            lines[k] = ""
        if indent:
            lines[i] = indent + "pass  # kernel_closure_check: Python import removed"
        i = j + 1
    return "\n".join(lines), removed


def module_name(path):
    """Dotted import name of a .mojo file, or (None, reason) when Mojo cannot import it as a module."""
    parts = path[:-len(".mojo")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            return None, "root __init__.mojo"
    for seg in parts:
        if not IDENT.match(seg) or keyword.iskeyword(seg):
            return None, f"path segment {seg!r} is not an identifier"
    return ".".join(parts), None  # Mojo resolves a directory on -I without an __init__.mojo too


def _resolve_module(dotted, paths):
    base = dotted.replace(".", "/")
    for cand in (base + ".mojo", base + "/__init__.mojo"):
        if cand in paths:
            return cand
    return None


def imports_of(path, text, paths):
    """Repository .mojo files that `path` imports (standard-library imports are not files here)."""
    out = set()
    lines = text.split("\n")
    pkg = path.split("/")[:-1]
    i = 0
    while i < len(lines):
        ln = lines[i].split("#", 1)[0]
        m = FROM_IMPORT.match(ln)
        if m:
            src, names = m.group(1), m.group(2)
            while names.count("(") > names.count(")") and i + 1 < len(lines):
                i += 1
                names += " " + lines[i].split("#", 1)[0]
            dots = len(src) - len(src.lstrip("."))
            rest = src[dots:]
            if dots:
                up = pkg[:len(pkg) - (dots - 1)] if dots - 1 <= len(pkg) else []
                dotted = ".".join(up + ([rest] if rest else []))
            else:
                dotted = rest
            hit = _resolve_module(dotted, paths) if dotted else None
            if hit:
                out.add(hit)
            for nm in re.split(r"[,()\s]+", names):
                if IDENT.match(nm) and nm != "as":
                    sub = _resolve_module((dotted + "." if dotted else "") + nm, paths)
                    if sub:
                        out.add(sub)
        else:
            m = PLAIN_IMPORT.match(ln)
            if m:
                for item in m.group(1).split(","):
                    dotted = item.strip().split(" as ")[0].strip()
                    hit = _resolve_module(dotted, paths) if dotted else None
                    if hit:
                        out.add(hit)
        i += 1
    return out


def python_reach(texts, edges):
    """Files that import or call Python, or import such a file (transitively)."""
    reach = {p for p, t in texts.items() if PY_LINE_M.search(t) or PY_CALL.search(t)}
    changed = True
    while changed:
        changed = False
        for p, deps in edges.items():
            if p not in reach and deps & reach:
                reach.add(p)
                changed = True
    return reach


def plan(texts):
    paths = set(texts)
    kernels = sorted(p for p in paths if not is_shell(p) and not is_tooling(p))
    modules, skipped, mains = [], [], []
    for p in kernels:
        if MAIN_DEF.search(texts[p]):
            mains.append(p)
            continue
        name, why = module_name(p)
        if name is None:
            skipped.append({"path": p, "reason": why})
        else:
            modules.append((p, name))
    edges = {p: imports_of(p, t, paths) for p, t in texts.items()}
    reach = python_reach(texts, edges)
    py_import = [p for p in kernels if PY_LINE_M.search(texts[p])]
    py_call = [p for p in kernels if PY_CALL.search(texts[p])]
    door, shell_helpers = [], []
    for p in kernels:
        for d in sorted(edges[p]):
            if d in reach and (is_shell(d) or is_tooling(d)):
                door.append({"kernel": p, "imports": d, "kind": "shell" if is_shell(d) else "tooling"})
            elif is_shell(d):
                shell_helpers.append({"kernel": p, "imports": d})
    return {"kernels": kernels, "modules": modules, "skipped": skipped, "mains": mains, "reach": reach,
            "py_import": py_import, "py_call": py_call, "door": door, "shell_helpers": shell_helpers}


def program_text(sha, modules):
    lines = [f"# Generated by tools/kernel_closure_check.py from {sha[:12]}; do not edit or commit.",
             "# Imports every importable kernel module (no shells, no tooling, no programs with main).",
             "# The check builds it in a tree with every shell deleted and every Python import removed."]
    lines += [f"import {name} as _k{i:04d}" for i, (_p, name) in enumerate(modules)]
    lines += ["", "", "def main():", f'    print("kernel closure: {len(modules)} kernel modules imported, no Python")', ""]
    return "\n".join(lines)


def write_tree(root, texts, repo, reach):
    """Write the stripped tree; returns (stripped files, deleted shells, generated-file notes)."""
    if root.exists():
        shutil.rmtree(root)
    stripped, deleted = {}, []
    for p, t in sorted(texts.items()):
        if is_shell(p) and p in reach:
            deleted.append(p)
            continue
        body, n = strip_python(t)
        if n:
            stripped[p] = n
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    notes = []
    for rel, cmd in GENERATED.items():
        src = repo / rel
        if not src.exists() and (repo / cmd[-1]).exists():
            r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
            notes.append({"file": rel, "command": " ".join(cmd), "exit_code": r.returncode,
                          "stderr_tail": r.stderr[-400:]})
        if src.exists():
            body, n = strip_python(src.read_text(encoding="utf-8"))
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(body, encoding="utf-8")
        else:
            notes.append({"file": rel, "missing": True, "make_it_with": " ".join(cmd)})
    return stripped, deleted, notes


def run(cmd, log, timeout, cwd=None):
    start = time.monotonic()
    with open(log, "w") as f:
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, cwd=cwd)
            code, timed_out = r.returncode, False
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    return {"command": cmd, "exit_code": code, "timeout": timed_out, "seconds": round(time.monotonic() - start, 1),
            "log": str(log)}


def inspect_binary(binary, out):
    errors = []
    dep_cmd = ["otool", "-L", str(binary)] if sys.platform == "darwin" else ["ldd", str(binary)]
    r = subprocess.run(dep_cmd, capture_output=True, text=True)
    (out / "dependencies.log").write_text(r.stdout + r.stderr)
    if r.returncode and "not a dynamic executable" not in (r.stdout + r.stderr):
        errors.append("dependency inspection failed")
    if PYTHON_LIB.search(r.stdout):
        errors.append("libpython dependency")
    r = subprocess.run(["nm", "-u", str(binary)], capture_output=True, text=True)
    (out / "symbols.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        errors.append("undefined-symbol inspection failed")
    syms = sorted(set(PYTHON_SYMBOL.findall(r.stdout)))
    if syms:
        errors.append("undefined CPython symbols: " + ", ".join(syms[:10]))
    return errors


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--out", default="kernel-closure-results")
    ap.add_argument("--mojo", default="mojo")
    ap.add_argument("--timeout", type=float, default=2700)
    ap.add_argument("--emit-only", action="store_true")
    args = ap.parse_args(argv)
    top = git("rev-parse", "--show-toplevel").stdout.decode().strip()
    if not top:
        print("kernel_closure_check: run inside the mojolearn checkout", file=sys.stderr)
        return 2
    repo = Path(top)
    sha = git("rev-parse", "--verify", "--quiet", args.ref + "^{commit}", cwd=repo).stdout.decode().strip()
    if not sha:
        print(f"kernel_closure_check: unknown ref {args.ref}", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    texts = read_tree(sha, repo)
    pl = plan(texts)
    program = program_text(sha, pl["modules"])
    tree = out / "tree"
    stripped, deleted, notes = write_tree(tree, texts, repo, pl["reach"])
    (tree / PROGRAM_REL).parent.mkdir(parents=True, exist_ok=True)
    (tree / PROGRAM_REL).write_text(program, encoding="utf-8")
    (out / "kernel_closure_main.mojo").write_text(program, encoding="utf-8")
    in_repo = repo / PROGRAM_REL
    in_repo.parent.mkdir(parents=True, exist_ok=True)
    in_repo.write_text(program, encoding="utf-8")
    binary = out / "kernel_closure"
    build_cmd = [args.mojo, "build", *FLAGS, "-I", str(tree), str(tree / PROGRAM_REL), "-o", str(binary)]
    summary = {"commit": sha, "kernel_files": len(pl["kernels"]), "modules": len(pl["modules"]),
               "programs_with_main": pl["mains"], "not_importable": pl["skipped"],
               "kernel_python_imports": pl["py_import"], "kernel_python_calls": pl["py_call"],
               "kernel_imports_of_the_door": pl["door"], "stripped_files": stripped,
               "deleted_shells": deleted, "kernel_imports_of_python_free_shell_files": pl["shell_helpers"],
               "generated_files": notes, "build_command": build_cmd}
    head = (f"kernel_closure_check: {sha[:9]}: {len(pl['kernels'])} kernel files, {len(pl['modules'])} imported, "
            f"{len(pl['mains'])} programs with main, {len(pl['skipped'])} not importable; "
            f"{len(pl['py_import'])} import Python, {len(pl['py_call'])} call external_call[\"Py, "
            f"{len(pl['door'])} import a shell or tooling file that reaches Python; {len(deleted)} shells deleted, "
            f"Python imports removed from {len(stripped)} files; {len(pl['shell_helpers'])} kernel imports of "
            f"Python-free shell files (helpers to move out of the shell)")
    print(head, file=sys.stderr)
    for d in pl["door"][:20]:
        print(f"  {d['kernel']} imports {d['imports']} ({d['kind']})", file=sys.stderr)
    for p in pl["py_call"][:20]:
        print(f"  {p}: external_call[\"Py...\"] (cannot be stripped)", file=sys.stderr)
    for n in notes:
        print(f"  generated file {n}", file=sys.stderr)
    if args.emit_only:
        summary["status"] = "emitted"
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print("kernel_closure_check: build with: " + " ".join(shlex.quote(c) for c in build_cmd), file=sys.stderr)
        print(f"KERNEL_CLOSURE status=emitted modules={len(pl['modules'])} program={in_repo}", file=sys.stderr)
        print(in_repo)
        return 0
    if not shutil.which(args.mojo) and not Path(args.mojo).exists():
        print(f"kernel_closure_check: no {args.mojo} on PATH (run under pixi run)", file=sys.stderr)
        return 2
    build = run(build_cmd, out / "build.log", args.timeout, cwd=tree)
    summary["build"] = build
    errors = []
    if build["timeout"]:
        errors.append(f"compile exceeded {args.timeout:.0f} s")
    elif build["exit_code"]:
        errors.append(f"compile failed (exit {build['exit_code']}); first errors in {out / 'build.log'}")
    else:
        errors += inspect_binary(binary, out)
        r = run([str(binary)], out / "run.log", 120)
        summary["run"] = r
        if r["exit_code"] != 0:
            errors.append(f"program exit {r['exit_code']}")
    if pl["py_call"]:
        errors.append(f"{len(pl['py_call'])} kernel files call external_call[\"Py")
    if pl["door"]:
        errors.append(f"{len(pl['door'])} kernel imports of a shell or tooling file that reaches Python")
    summary["errors"] = errors
    summary["status"] = "fail" if errors else "pass"
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for e in errors:
        print(f"kernel_closure_check: {e}", file=sys.stderr)
    print(f"KERNEL_CLOSURE status={summary['status']} modules={len(pl['modules'])} stripped={len(stripped)} "
          f"door={len(pl['door'])} py_calls={len(pl['py_call'])} build_rc={build['exit_code']} "
          f"build_s={build['seconds']} summary={out / 'summary.json'}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
