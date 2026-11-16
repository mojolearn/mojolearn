#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""LAST-USE AUDIT: a local whose memory a pointer still reads after Mojo has freed the local.

    python3 tools/mojo_lastuse_audit.py [--commit <sha>] [--root DIR] [--all-kinds]

Mojo destroys a local at its LAST NAMED USE. A pointer derived from it keeps it alive only when the pointer's origin
says so. Mojo 1.2.0 stdlib, `std/origin/__init__.mojo` (tag mojo/v1.2.0):
  * a tracked origin (`x.unsafe_ptr()`, `Pointer(to=x)`, `Span(x)`, `.bitcast`) is `origin_of(x)`: x lives as long
    as the pointer does;
  * `MutAnyOrigin`/`ImmutAnyOrigin` are `#lit.any.origin`, the same attribute as `UnsafeAnyOrigin`, whose docstring
    says "Every other value in scope is kept alive for as long as the reference is live ... effectively halting ASAP
    destruction" (also the manual, mojolang.org/docs/manual/values/lifetimes, "Wildcard origins"): x lives as long as
    the pointer does;
  * `MutUntrackedOrigin`/`ImmUntrackedOrigin` are the EMPTY origin ("nothing for the lifetime checker to track or
    extend"), and an address (`Int(x.unsafe_ptr())`, `unsafe_from_address=`, a helper returning `Int(arg.unsafe_ptr())`)
    has no origin at all: x dies at its last named use, whatever the pointer still does.
So the audit fails on an UNTRACKED derivation of a local X where
  * the derivation is X's last named use and sits inside a call's arguments (X is freed before the call reads it:
    xtrees/calibrated_apply.mojo, the five native check suites of 2026-11-16), or is returned; or
  * it is assigned to P and P (or a name assigned from P) is used after X's last named use.
Any-origin and tracked derivations are counted (`--all-kinds` lists them) and never fail.

Fix: name the local again after the pointer's last use (`_ = x^`; free). A true false alarm (a stack array whose
slot outlives the name, a value the caller owns) gets one line in tools/mojo_lastuse_allow.txt:
`<path>:<function>:<local>  # <reason>`. Unlisted findings fail; allow lines that match nothing are printed as STALE.

Every tracked .mojo file (git ls-files, or the blobs of --commit) is scanned, split across os.cpu_count() worker
processes. tools/prebuild_check.py runs it in its `audits` part (seconds).
"""
import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOW = "tools/mojo_lastuse_allow.txt"
_DEF = re.compile(r"^(\s*)def\s+(\w+)")
_VAR = re.compile(r"(?<![\w.])var\s+(\w+)\b")
_PARAM_VAR = re.compile(r"[(,]\s*(?:var|owned)\s+(\w+)\s*:")
_ASSIGN = re.compile(r"^\s*(?:var\s+)?(\w+)\s*(?::[^=]*)?=(?!=)")
_HELPER_DEF = re.compile(r"^\s*def\s+(\w+)\s*\(([^)]*)\)\s*(?:raises\s*)?->\s*(\w+)")
_HELPER_RET = r"return\s+(?:U?Int\(\s*{p}\.unsafe_ptr\(\)|{p}\.unsafe_ptr\(\)[^\n]*UntrackedOrigin)"
#: wrappers that turn their argument into an address or pointer; not calls that consume the pointer
_CONVERT = {"Int", "UInt", "UnsafePointer", "Pointer", "MutPointer", "ImmPointer", "OpaquePointer", "rebind"}
_IDENT = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\b(?!=[^=])")
_HOT = re.compile(r"unsafe_ptr|unsafe_origin_cast|as_unsafe_any_origin|\bto=")
_READ = re.compile(r"\.(?:unsafe_ptr|unsafe_origin_cast\[[^\]]*\]|bitcast\[[^\]]*\])\(\)\s*(?:\[|\.(?:unsafe_load|load|unsafe_get)\b)")
_KEEP = re.compile(r"^\s*_\s*=\s*\w+\^?\s*$")
_LOOP = re.compile(r"^\s*(?:for|while)\b")
_STRING = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def _strip(line):
    """LINE without string literals and its comment."""
    line = _STRING.sub('""', line)
    cut = line.find("#")
    return line if cut < 0 else line[:cut]


def statements(lines):
    """[(first line index, last line index, code)] of LINES: comments, strings and docstrings removed, a statement
    continued over lines while a bracket is open."""
    out, buf, start, depth, in_doc = [], [], None, 0, None
    for i, raw in enumerate(lines):
        if in_doc:
            if in_doc in raw:
                in_doc = None
            continue
        s = raw.strip()
        doc = next((q for q in ('"""', "'''") if s.startswith(q) and not buf), None)
        if doc:  # a docstring (or a bare string block): skipped, closed on its line or a later one
            if s.count(doc) == 1:
                in_doc = doc
            continue
        code = _strip(raw)
        if not code.strip() and not buf:
            continue
        if start is None:
            start = i
        buf.append(code)
        depth += sum(code.count(c) for c in "([{") - sum(code.count(c) for c in ")]}")
        if depth <= 0:
            out.append((start, i, " ".join(b.strip() for b in buf)))
            buf, start, depth = [], None, 0
    if buf:
        out.append((start, len(lines) - 1, " ".join(b.strip() for b in buf)))
    return out


def functions(lines):
    """[(name, def line index, end line index exclusive)] for every def, nested ones included (an outer body
    contains its nested defs: a closure's capture is a use)."""
    defs = []
    for i, line in enumerate(lines):
        m = _DEF.match(line)
        if not m:
            continue
        indent, end = len(m.group(1)), len(lines)
        j = i + 1
        # skip the rest of a multi-line signature
        depth = line.count("(") - line.count(")")
        while depth > 0 and j < len(lines):
            depth += lines[j].count("(") - lines[j].count(")")
            j += 1
        for k in range(j, len(lines)):
            t = lines[k]
            if t.strip() and not t.lstrip().startswith("#") and len(t) - len(t.lstrip()) <= indent:
                end = k
                break
        defs.append((m.group(2), i, end))
    return defs


def helpers_of(text):
    """{name: 'untracked'} for the defs in TEXT that return an address or untracked pointer of a parameter, and
    {'alias:NAME': 'any'} for its pointer aliases with a wildcard origin (`comptime _AP = MutPointer[T, MutAnyOrigin]`)."""
    found = {}
    for m in re.finditer(r"^comptime\s+(\w+)\s*=\s*(?:Mut|Imm)?(?:Unsafe)?Pointer\[[^\n]*AnyOrigin", text, re.M):
        found["alias:" + m.group(1)] = "any"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _HELPER_DEF.match(line)
        if not m:
            continue
        params = [p.split(":")[0].split()[-1] for p in m.group(2).split(",") if ":" in p]
        body = "\n".join(lines[i + 1:i + 8])
        if any(re.search(_HELPER_RET.format(p=re.escape(p)), body) for p in params):
            found[m.group(1)] = "untracked"
    return found


def _chain_end(s, k):
    """End of the postfix chain (.name, [..], (..)) that starts at S[k]."""
    while k < len(s):
        if s[k] == "." and k + 1 < len(s) and (s[k + 1].isalpha() or s[k + 1] == "_"):
            k += 1
            while k < len(s) and (s[k].isalnum() or s[k] == "_"):
                k += 1
        elif s[k] in "[(":
            close, depth = {"[": "]", "(": ")"}[s[k]], 0
            open_ = s[k]
            while k < len(s):
                depth += (s[k] == open_) - (s[k] == close)
                k += 1
                if depth == 0:
                    break
        else:
            break
    return k


def _enclosing_calls(s, p):
    """Names before each open '(' that encloses position P (innermost first). A pointer constructor
    (`T(unsafe_from_address=...)`, T an alias) is named `UnsafePointer`."""
    names, depth = [], 0
    for k in range(p - 1, -1, -1):
        c = s[k]
        if c in ")]":
            depth += 1
        elif c in "([":
            if depth:
                depth -= 1
                continue
            if c == "(":
                m = re.search(r"([\w.]+)(?:\[[^\[\]]*\])?\s*$", s[:k])
                ctor = s[k + 1:].lstrip().startswith("unsafe_from_address=")
                names.append("UnsafePointer" if ctor else (m.group(1).split(".")[-1] if m else ""))
    return names


def _arg_span(s, p):
    """The text of the argument (of the innermost enclosing call that is not a conversion) that holds position P:
    an any-origin pointer built around an address (`MutPointer[T, MutAnyOrigin](unsafe_from_address=Int(..))`)
    keeps the value alive like any other any-origin pointer."""
    lo, depth = 0, 0
    for k in range(p - 1, -1, -1):
        c = s[k]
        if c in ")]":
            depth += 1
        elif c in "([":
            if depth:
                depth -= 1
                continue
            m = re.search(r"([\w.]+)(?:\[[^\[\]]*\])?\s*$", s[:k]) if c == "(" else None
            ctor = s[k + 1:].lstrip().startswith("unsafe_from_address=")
            if c == "(" and not ctor and (m.group(1).split(".")[-1] if m else "") not in _CONVERT:
                lo = k + 1
                break
        elif c == "," and depth == 0:
            lo = k + 1
            break
    hi, depth = len(s), 0
    for k in range(p, len(s)):
        c = s[k]
        if c in "([":
            depth += 1
        elif c in ")]":
            if depth == 0:
                hi = k
                break
            depth -= 1
        elif c == "," and depth == 0:
            hi = k
            break
    return s[lo:hi]


def derivation(s, name, helpers):
    """(kind, sink) of the first pointer derivation of NAME in statement S, or None. kind: untracked, any,
    tracked; sink: call, return, assign. A pointer read on the spot (`x.unsafe_ptr()[i]`, `.unsafe_load(`) and a
    pointer into what X points at (`x[].field.unsafe_ptr()`: X is a pointer, not the owner) are not derivations."""
    for m in re.finditer(r"(?<![\w.])" + re.escape(name) + r"\b(?!=)", s):
        p, kind = m.start(), None
        before = s[:p]
        if s[m.end():].startswith("[]"):
            continue
        hm = re.search(r"\b(\w+)\(\s*$", before)
        if hm and helpers.get(hm.group(1)) == "untracked" and s[m.end():].lstrip().startswith(")"):
            kind, start = helpers[hm.group(1)], hm.start(1)
        else:
            to = re.search(r"\b(?:Unsafe)?Pointer(?:\[[^\[\]]*\])?\(\s*to=$", before)
            start = to.start() if to else p
            end = _chain_end(s, s.index(")", m.end()) + 1) if to and ")" in s[m.end():] else _chain_end(s, m.end())
            chain = s[start:end]
            # `x.unsafe_origin_cast[...]` alone: X is already a pointer and owns nothing
            if not to and not re.search(r"\.unsafe_ptr\b", chain):
                continue
            if _READ.search(chain):
                continue
            arg = _arg_span(s, start)
            for alias, akind in helpers.items():
                if alias.startswith("alias:") and re.search(r"\b" + alias[6:] + r"\b", arg):
                    arg += " " + ("AnyOrigin" if akind == "any" else "")
            if "UntrackedOrigin" in chain:
                kind = "untracked"
            elif re.search(r"AnyOrigin|as_unsafe_any_origin", arg):
                kind = "any"
            elif re.search(r"(?:\bU?Int\(|unsafe_from_address=)\s*$", s[:start]):
                kind = "untracked"
            else:
                kind = "tracked"
        calls = [c for c in _enclosing_calls(s, start) if c not in _CONVERT]
        if calls:
            sink = "call"
        elif re.match(r"^\s*return\b", s):
            sink = "return"
        else:
            sink = "assign"
        return kind, sink
    return None


def scan_text(path, text, helpers):
    """[finding dict] of one file."""
    lines = text.splitlines()
    stmts = statements(lines)
    out, seen = [], set()
    for fname, a, b in functions(lines):
        body = [(i, j, s) for (i, j, s) in stmts if a < i < b or (i == a)]
        if not body:
            continue
        indent = [len(lines[i]) - len(lines[i].lstrip()) for (i, _, _) in body]
        n_body = len(body)

        def block_end(k):
            """First statement after K's block (the statements indented deeper than K that follow it)."""
            for q in range(k + 1, n_body):
                if indent[q] <= indent[k]:
                    return q
            return n_body

        def scope_end(k):
            """End of the scope a `var` declared at K lives in: the first later statement indented less."""
            for q in range(k + 1, n_body):
                if indent[q] < indent[k]:
                    return q
            return n_body

        idents = [set(_IDENT.findall(s)) if k else set() for k, (_, _, s) in enumerate(body)]
        calls = [h for h, v in helpers.items() if v == "untracked"]
        hot = [bool(_HOT.search(s)) or any(h + "(" in s for h in calls) for (_, _, s) in body]
        loops = [k for k, (_, _, s) in enumerate(body) if _LOOP.match(s)]
        decls = {}  # name -> [(declaration statement, scope end)]
        for v in _PARAM_VAR.findall(body[0][2]):
            decls.setdefault(v, []).append((0, n_body))
        for k in range(1, n_body):
            for v in _VAR.findall(body[k][2]):
                decls.setdefault(v, []).append((k, scope_end(k)))
        decls.pop("self", None)

        def uses(n, lo, hi):
            return [k for k in range(max(lo, 1), hi) if n in idents[k]]

        def scope_of(n, k):
            """(declaration, scope end) of the NAME visible at statement K, or None."""
            best = None
            for d, e in decls.get(n, ()):
                if d <= k < e and (best is None or d >= best[0]):
                    best = (d, e)
            return best

        def effective_last(d, last):
            """LAST extended to the end of every loop that encloses it and began after the declaration D: the
            value lives across the back edge."""
            eff = last
            for h in loops:
                if d < h <= last < block_end(h):
                    eff = max(eff, block_end(h) - 1)
            return eff

        for x, scopes in sorted(decls.items()):
            for d, e in scopes:
                ux = uses(x, d, e)
                if not ux:
                    continue
                last_x = ux[-1]
                moved = re.search(r"(?<![\w.])" + re.escape(x) + r"\^", body[last_x][2]) and not re.match(
                    r"^\s*_\s*=\s*" + re.escape(x) + r"\^\s*$", body[last_x][2])
                eff_x = effective_last(d, last_x)
                for k in ux:
                    if not hot[k]:
                        continue
                    s = body[k][2]
                    found = derivation(s, x, helpers)
                    if not found:
                        continue
                    kind, sink = found
                    bad, why = False, ""
                    if sink == "assign":
                        am = _ASSIGN.match(s)
                        if am and am.group(1) != x:
                            names, frontier, p_last = {am.group(1)}, [am.group(1)], k
                            while frontier:  # names assigned from the pointer after the derivation
                                n = frontier.pop()
                                sc = scope_of(n, k) or (d, e)
                                for q in uses(n, k + 1, sc[1]):
                                    if _KEEP.match(body[q][2]):
                                        continue  # `_ = p`: a keep-alive, not a read through the pointer
                                    p_last = max(p_last, q)
                                    bm = _ASSIGN.match(body[q][2])
                                    if bm and bm.group(1) not in names and bm.group(1) not in (n, "_") and \
                                            _carries(body[q][2], n):
                                        names.add(bm.group(1))
                                        frontier.append(bm.group(1))
                            if p_last > eff_x:
                                bad, why = True, f"pointer {am.group(1)} used at line {body[p_last][0] + 1}"
                    elif k == last_x and eff_x == last_x:
                        bad, why = True, ("freed before the call reads it" if sink == "call" else "returned")
                    if kind != "untracked" or moved:
                        bad = False
                    key = (x, body[k][0])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(dict(path=path, line=body[k][0] + 1, function=fname, local=x, kind=kind, sink=sink,
                                    last_use=body[last_x][0] + 1, fails=bad, why=why))
    return out


def _carries(s, n):
    """Whether assignment S copies the pointer N into its target (`q = n + 4`, `[n, m]`,
    `MutPointer(unsafe_from_address=n)`), not a value a call computed from it."""
    rhs = s[_ASSIGN.match(s).end():]
    off = len(s) - len(rhs)
    for m in re.finditer(r"(?<![\w.])" + re.escape(n) + r"\b", rhs):
        if re.match(r"\s*(?:\[|\.(?:unsafe_load|load|unsafe_get)\b)", rhs[m.end():]):
            continue  # a value read through the pointer, not the pointer
        if not [c for c in _enclosing_calls(s, off + m.start()) if c not in _CONVERT]:
            return True
    return False


def _scan_many(items):
    return [f for path, text, helpers in items for f in scan_text(path, text, helpers)]


def _helper_pass(items):
    return [(path, helpers_of(text)) for path, text in items]


def read_files(root=ROOT, commit=None):
    """[(path, text)] of every tracked .mojo file in ROOT's work tree, or of COMMIT's blobs."""
    if commit is None:
        names = subprocess.run(["git", "ls-files", "-z", "--", "*.mojo"], cwd=str(root), capture_output=True,
                               check=True).stdout.decode().split("\0")
        return [(n, (Path(root) / n).read_text(errors="replace")) for n in names if n and (Path(root) / n).is_file()]
    listing = subprocess.run(["git", "ls-tree", "-r", "-z", commit], cwd=str(root), capture_output=True,
                             check=True).stdout.decode().split("\0")
    rows = [r.split("\t", 1) for r in listing if r.endswith(".mojo") and " blob " in r]
    shas = [(meta.split()[2], path) for meta, path in rows]
    proc = subprocess.run(["git", "cat-file", "--batch"], cwd=str(root), check=True, capture_output=True,
                          input="".join(f"{sha}\n" for sha, _ in shas).encode())
    data, pos, out = proc.stdout, 0, []
    for _, path in shas:
        header_end = data.index(b"\n", pos)
        size = int(data[pos:header_end].split()[2])
        out.append((path, data[header_end + 1:header_end + 1 + size].decode(errors="replace")))
        pos = header_end + 1 + size + 1
    return out


def _module_file(path, module):
    """The repo path of MODULE imported from PATH (`a.b` -> a/b.mojo; `.b` relative to PATH's directory)."""
    if module.startswith("."):
        base = Path(path).parent
        dots = len(module) - len(module.lstrip("."))
        for _ in range(dots - 1):
            base = base.parent
        rel = base / module.lstrip(".").replace(".", "/")
    else:
        rel = Path(module.replace(".", "/"))
    return rel.as_posix() + ".mojo"


def _chunks(items, n):
    return [items[i::n] for i in range(n) if items[i::n]]


def scan(files, workers=None):
    """[finding] over FILES [(path, text)], split across WORKERS processes (default os.cpu_count())."""
    workers = max(1, workers or os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        per_file = dict(h for part in pool.map(_helper_pass, _chunks(files, workers)) for h in part)
        items = []
        for path, text in files:
            local = dict(per_file.get(path, {}))
            for m in re.finditer(r"^\s*from\s+(\S+)\s+import\s+([^\n]+)", text, re.M):
                source = per_file.get(_module_file(path, m.group(1)), {})
                for name in re.findall(r"\w+", m.group(2)):
                    if source.get(name) == "untracked":
                        local.setdefault(name, "untracked")
                    if "alias:" + name in source:
                        local.setdefault("alias:" + name, source["alias:" + name])
            items.append((path, text, local))
        found = [f for part in pool.map(_scan_many, _chunks(items, workers)) for f in part]
    return sorted(found, key=lambda f: (f["path"], f["line"], f["local"]))


def load_allow(text):
    """{(path, function, local): reason} of an allow-list; a line without a reason is an error."""
    allow, errors = {}, []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, reason = line.partition("#")
        parts = key.strip().split(":")
        if len(parts) != 3 or not reason.strip():
            errors.append(f"{ALLOW}:{n}: want '<path>:<function>:<local>  # <reason>'")
            continue
        allow[tuple(parts)] = reason.strip()
    return allow, errors


def verdict(found, allow):
    """(failing findings, allow-listed findings, stale allow keys)."""
    failing, allowed, used = [], [], set()
    for f in found:
        if not f["fails"]:
            continue
        key = (f["path"], f["function"], f["local"])
        if key in allow:
            allowed.append(f)
            used.add(key)
        else:
            failing.append(f)
    return failing, allowed, sorted(set(allow) - used)


def audit(root=ROOT, commit=None, workers=None, all_kinds=False, out=print):
    """Run the audit; print lines; return the exit code."""
    files = read_files(root, commit)
    if commit is None:
        p = Path(root) / ALLOW
        allow_text = p.read_text() if p.is_file() else ""
    else:
        r = subprocess.run(["git", "show", f"{commit}:{ALLOW}"], cwd=str(root), capture_output=True, text=True)
        allow_text = r.stdout if r.returncode == 0 else ""
    allow, errors = load_allow(allow_text)
    found = scan(files, workers)
    failing, allowed, stale = verdict(found, allow)
    kinds = {}
    for f in found:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    for e in errors:
        out(f"LASTUSE FAIL {e}")
    for f in failing:
        out(f"LASTUSE FAIL {f['path']}:{f['line']}: {f['local']} in {f['function']} ({f['kind']} pointer, "
            f"{f['sink']}): last named use line {f['last_use']}, {f['why']}; keep it alive (`_ = {f['local']}^`)")
    for key in stale:
        out(f"LASTUSE STALE allow line matches nothing: {':'.join(key)}")
    if all_kinds:
        for f in found:
            if not f["fails"]:
                out(f"LASTUSE INFO {f['path']}:{f['line']}: {f['local']} ({f['kind']}, {f['sink']})")
    bad = len(failing) + len(errors)
    out(f"LASTUSE {'FAIL' if bad else 'PASS'} {len(files)} .mojo files, derivations {dict(sorted(kinds.items()))}, "
        f"{len(failing)} unfixed, {len(allowed)} allow-listed, {len(stale)} stale allow line(s)")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", default=None, help="scan this commit's blobs instead of the work tree")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--workers", type=int, default=None, help="default os.cpu_count()")
    ap.add_argument("--all-kinds", action="store_true", help="also list the tracked and any-origin derivations")
    a = ap.parse_args(argv)
    return audit(Path(a.root), a.commit, a.workers, a.all_kinds)


if __name__ == "__main__":
    sys.exit(main())
