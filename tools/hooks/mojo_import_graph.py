#!/usr/bin/env python3
"""Mojo imports at a Git revision, used only to estimate rebuild scope.

This has no runtime-route classification or compliance verdict. Git's batch
object stream is ordered I/O; independent source parsing uses all available
CPU cores. Workers share the read-only blob store through fork, matching the
existing local hook process model.
"""
import collections
import concurrent.futures
import multiprocessing
import os
import re
import subprocess

_BLOB_CACHE = {}
_IMPORT_FROM = re.compile(r"^\s*from\s+(\.*[\w.]*)\s+import\s+(.*)$")
_IMPORT = re.compile(r"^\s*import\s+(.*)$")

def _git(*args, inp=None):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          errors="replace", input=inp)

def _ls(ref):
    """{path: blob sha} of every file at ref."""
    r = _git("ls-tree", "-r", "--full-tree", ref)
    if r.returncode != 0:
        raise SystemExit(f"mojo_import_graph: cannot list {ref}: {r.stderr.strip()}")
    out = {}
    for ln in r.stdout.splitlines():
        meta, _, path = ln.partition("\t")
        bits = meta.split()
        if len(bits) == 3 and bits[1] == "blob":
            out[path] = bits[2]
    return out

def _read_blobs(shas):
    """{sha: text}, one `git cat-file --batch` process for the uncached ones."""
    need = [s for s in dict.fromkeys(shas) if s not in _BLOB_CACHE]
    if need:
        p = subprocess.run(["git", "cat-file", "--batch"], input="".join(s + "\n" for s in need).encode(),
                           capture_output=True)
        out, i = p.stdout, 0
        for sha in need:
            nl = out.index(b"\n", i)
            head = out[i:nl].split()
            i = nl + 1
            if len(head) < 3 or head[1] == b"missing":
                _BLOB_CACHE[sha] = ""
                continue
            size = int(head[2])
            _BLOB_CACHE[sha] = out[i:i + size].decode("utf-8", "replace")
            i += size + 1
    return {s: _BLOB_CACHE[s] for s in shas}

def _imports(lines):
    """[(line_no, module, [(name, alias)])] for each import statement,
    joining parenthesized continuation lines."""
    res = []
    i = 0
    while i < len(lines):
        no, ln = lines[i]
        m = _IMPORT_FROM.match(ln)
        if m:
            mod, rest = m.group(1), m.group(2)
            if "(" in rest and ")" not in rest:
                j = i + 1
                while j < len(lines) and ")" not in lines[j][1]:
                    rest += " " + lines[j][1]
                    j += 1
                if j < len(lines):
                    rest += " " + lines[j][1]
                i = j
            names = []
            for part in rest.replace("(", " ").replace(")", " ").split(","):
                part = part.split("#")[0].strip()
                if not part:
                    continue
                bits = part.split()
                name = bits[0]
                alias = bits[2] if len(bits) >= 3 and bits[1] == "as" else name
                names.append((name, alias))
            res.append((no, mod, names))
        else:
            m = _IMPORT.match(ln)
            if m:
                for part in m.group(1).split(","):
                    bits = part.strip().split()
                    if bits:
                        alias = bits[2] if len(bits) >= 3 and bits[1] == "as" else bits[0]
                        res.append((no, bits[0], [("", alias)]))
        i += 1
    return res

def _resolve(mod, src, files):
    """Repo paths of a Mojo import's module (and the submodule a name may
    be), searching the package root and bindings/ (the build's -I paths)."""
    if mod.startswith("."):
        dots = len(mod) - len(mod.lstrip("."))
        base = os.path.dirname(src)
        for _ in range(dots - 1):
            base = os.path.dirname(base)
        rel = mod.lstrip(".").replace(".", "/")
        cands = [os.path.join(base, rel) if rel else base]
    else:
        rel = mod.replace(".", "/")
        cands = [rel, "bindings/" + rel]
    out = []
    for c in cands:
        c = os.path.normpath(c)
        for p in (c + ".mojo", c + "/__init__.mojo"):
            if p in files:
                out.append(p)
        if out:
            # the packages along the path load too
            parts = c.split("/")
            for k in range(1, len(parts)):
                ini = "/".join(parts[:k]) + "/__init__.mojo"
                if ini in files:
                    out.append(ini)
            return out, c
    return out, None

def _code_lines(text):
    """Drop documentation and comment lines before reading imports."""
    result = []
    quote = None
    for number, line in enumerate(text.splitlines(), 1):
        if not quote and line.lstrip().startswith(("#", "//")):
            continue
        if quote:
            stop = line.find(quote)
            if stop < 0:
                continue
            line, quote = line[stop + 3:], None
        while True:
            found = re.search("\"\"\"|'''", line)
            if not found:
                break
            marker = found.group(0)
            stop = line.find(marker, found.end())
            if stop < 0:
                quote, line = marker, line[:found.start()]
                break
            line = line[:found.start()] + line[stop + 3:]
        if line.strip() and not line.lstrip().startswith(("#", "//")):
            result.append((number, line.rstrip()))
    return result


_FILES = set()


def _module_edges(item):
    path, sha = item
    edges = set()
    for _, module, names in _imports(_code_lines(_BLOB_CACHE[sha])):
        modules, base = _resolve(module, path, _FILES)
        edges.update(modules)
        if base is not None:
            for name, _ in names:
                if name:
                    child = module + name if module.endswith(".") else module + "." + name
                    edges.update(_resolve(child, path, _FILES)[0])
    return path, edges


class Tree:
    def __init__(self, ref):
        global _FILES
        shas = _ls(ref)
        self.files = set(shas)
        _FILES = self.files
        paths = sorted(p for p in shas if p.endswith(".mojo"))
        _read_blobs(list(dict.fromkeys(shas[p] for p in paths)))
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(cores, len(paths)) or 1,
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            self.edges = dict(pool.map(_module_edges, [(path, shas[path]) for path in paths]))
