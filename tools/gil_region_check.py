#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""No Python inside a GIL-released region of a Mojo door (lane meta-gil, 2026-11-23).

WHY. nn-integrate found verify --tamper-test segfaulting (CalibratedClassifierCV.score over a LogisticRegression
member: PyObject_GetAttrString) and Bagging OOB aborting (_PyMem_DebugMalloc called without the GIL). The members
were clean: bm-meta-3's doors evaluated `trees_address(acc)` (`acc._addr`: an attribute read and a new int) as an
argument of the RUN call INSIDE `with GILReleased(...)`. A static scan finds that class of defect before a build.

WHAT IT CHECKS. Every `with GILReleased(...):` block (the region is the lines indented deeper than the `with`)
of every .mojo file under bindings/ and providers/. Inside the region it flags:

  py-api      a Python API spelling: `py=`, `Python.`, `Python()`, `PythonObject(`, `py_get(` / `py_call(` /
              `py_item(` / `py_setitem(` / `py_type(` / `py_fields(` and the other `py_*` C-API helpers of
              bindings/py_balanced.mojo, `.__getattr__`, `cpython`/`CPython`;
  py-name     any use of a name the enclosing function holds as a PythonObject: a parameter or `var` typed
              `PythonObject`, a `var` assigned from `PythonObject(...)`, `Python.*`, a `py_*` helper or any function
              whose `-> PythonObject` return the scanned files declare (the file's own def wins; else every def of
              the name must agree), a subscript or attribute of another such name, or a `for` over one. The
              NEAREST `var` before the region types a re-declared name. A use there reads, copies (`_ = obj`
              copies and drops: two refcount changes) or hands the object to a callee that may read it;
  py-field    `<x>.<field>` where `<x>` is typed (a parameter, `var x: T`, `var x = T(...)` or a call whose
              return type is T) as a struct T that declares `var <field>: PythonObject`;
  py-return   a call of a function whose return type is `PythonObject`.

A finding is allowed only by a line `<path>:<function>:<token> <reason>` in tools/gil_region_allow.txt.

WHAT IT CANNOT SEE (say so in a report that leans on it): what a callee does with the plain values it is given
(a callee that takes Ints and reads Python through a global is invisible: the RUN modules are kept Python-free by
tools/python_free_dirs.sh instead); the type of a name inferred from a call whose return type it did not collect
or whose defs disagree (a method, an overload, a generic, a ternary): such a name is NOT flagged; an initializer
that is a comparison, or reads through `downcast_value_ptr` / `Int(py=...)`, is taken as a Mojo value; a struct held BY VALUE whose PythonObject field is
copied or destroyed by a whole-struct copy, move or scope exit inside the region (only `<x>.<field>` is seen);
regions opened by hand (`PyEval_SaveThread` / `PyEval_RestoreThread`, `PyGILState_*`) or by a helper that
releases inside (`release_gil=True` arguments); Mojo's last-use destruction of a Python-holding local whose last
named use falls inside the region through an alias (an Int address, a pointer).

Single process, no build, reads text only. Seconds over the whole tree.

    python3 tools/gil_region_check.py [--root DIR] [--commit SHA] [--allow FILE] [--verbose]
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ("bindings", "providers")
ALLOW = "tools/gil_region_allow.txt"

WITH_RE = re.compile(r"^(\s*)with\s+GILReleased\s*\(")
DEF_RE = re.compile(r"^(\s*)def\s+([A-Za-z_][A-Za-z_0-9]*)\s*[\[(]")
STRUCT_RE = re.compile(r"^(\s*)struct\s+([A-Za-z_][A-Za-z_0-9]*)")
FIELD_RE = re.compile(r"^\s+var\s+([A-Za-z_][A-Za-z_0-9]*)\s*:\s*PythonObject\b")
PARAM_RE = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*:\s*PythonObject\b")
VAR_ASSIGN_RE = re.compile(r"^\s*var\s+([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.+)$")
FOR_RE = re.compile(r"^\s*for\s+([A-Za-z_][A-Za-z_0-9]*)\s+in\s+([A-Za-z_][A-Za-z_0-9]*)\b")
PY_API_RE = re.compile(r"\bpy=|\bPython\s*\.|\bPython\s*\(\s*\)|\bPythonObject\s*\(|\bpy_[a-z_0-9]+\s*\(|"
                       r"__getattr__|\bcpython\b|\bCPython\b")
IDENT_RE = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z_0-9]*)\b")
#: an initializer that yields a Mojo value or pointer even when it starts from a PythonObject
NATIVE_VIEW_RE = re.compile(r"\.(downcast_value_ptr|unsafe_get_as_pointer|unchecked_downcast_value_ptr)\b|"
                            r"^\s*(Int|Float64|Float32|Bool|String|UInt8|Int32|Int64)\s*\(\s*py\s*=")
#: a comparison or identity test: the initializer is a Bool whatever its operands are
COMPARE_RE = re.compile(r"\bis\s+(not\s+)?None\b|\bis\s+[A-Za-z_]|==|!=|\band\b|\bor\b|\bnot\b")
FIELD_USE_RE = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z_0-9]*)\.([A-Za-z_][A-Za-z_0-9]*)\b")
PARAM_T_RE = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*:\s*([A-Z_][A-Za-z_0-9]*)\b")
VAR_T_RE = re.compile(r"^\s*var\s+([A-Za-z_][A-Za-z_0-9]*)\s*:\s*([A-Za-z_][A-Za-z_0-9]*)\b")
CALL_RE = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z_0-9]*)\s*\(")


def strip(line):
    """The line without its comment and string literals (a `#` inside a string is rare in doors; good enough)."""
    out, quote, i = [], None, 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "\"'":
            quote = c
            out.append('""')
        elif c == "#":
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


def indent(line):
    return len(line) - len(line.lstrip(" "))


def files_of(root, commit):
    """[(path, text)] of the .mojo files under DIRS, from the work tree ROOT or the git commit COMMIT (one
    `git cat-file --batch`)."""
    if commit is None:
        out = []
        for d in DIRS:
            for p in sorted((Path(root) / d).rglob("*.mojo")):
                out.append((str(p.relative_to(root)), p.read_text(encoding="utf-8", errors="replace")))
        return out
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit, "--", *DIRS], cwd=root,
                           capture_output=True, text=True, check=True).stdout.split()
    names = [n for n in names if n.endswith(".mojo")]
    batch = subprocess.run(["git", "cat-file", "--batch"], cwd=root, check=True, capture_output=True,
                           input="".join(f"{commit}:{n}\n" for n in names).encode())
    data, out, pos = batch.stdout, [], 0
    for n in names:
        nl = data.index(b"\n", pos)
        header = data[pos:nl].split()
        size = int(header[2]) if len(header) == 3 else 0
        out.append((n, data[nl + 1: nl + 1 + size].decode("utf-8", errors="replace")))
        pos = nl + 1 + size + 1
    return out


RET_RE = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z_0-9]*)\s*(?:\[[^\]]*\])?\s*\((?:[^()]|\([^()]*\))*\)"
                    r"\s*(?:raises\s*)?(?:->\s*([A-Za-z_][A-Za-z_0-9.]*))?", re.S)


def returns_of(text):
    """{function name: set of the return type names its defs in TEXT declare ('' for none)}."""
    out = {}
    for name, typ in RET_RE.findall(text):
        out.setdefault(name, set()).add(typ)
    return out


def collect(files):
    """(per-file return maps, global return map, {struct: its PythonObject field names}) over every file."""
    per_file, glob, fields = {}, {}, {}
    for path, text in files:
        local = returns_of(text)
        per_file[path] = local
        for name, types in local.items():
            glob.setdefault(name, set()).update(types)
        struct = None
        for line in text.splitlines():
            m = STRUCT_RE.match(line)
            if m:
                struct = (indent(line), m.group(2))
                continue
            if struct is not None and line.strip() and indent(line) <= struct[0] and not line.lstrip().startswith("#"):
                struct = None
            if struct is not None:
                f = FIELD_RE.match(line)
                if f:
                    fields.setdefault(struct[1], set()).add(f.group(1))
    return per_file, glob, fields


def return_type(name, local, glob):
    """The one return type name of function NAME: the file's own defs win; else every def of the name in the
    scanned files must agree; else None (not known)."""
    types = local.get(name) or glob.get(name)
    if types and len(types) == 1:
        return next(iter(types))
    return None


def expr_type(expr, names, types, local, glob):
    """'PythonObject', a struct name, or None for the value of EXPR (a `var` initializer)."""
    expr = expr.strip()
    if NATIVE_VIEW_RE.search(expr) or COMPARE_RE.search(expr):
        return None  # a Bool, a typed pointer or a Mojo value read out of a registered owner: not a PythonObject
    if re.match(r"(PythonObject|Python)\b", expr):
        return "PythonObject"
    m = re.match(r"([A-Za-z_][A-Za-z_0-9]*)\s*([\[.(])", expr)
    if not m:
        return None
    head, nxt = m.group(1), m.group(2)
    if head in names and nxt in "[.":
        return "PythonObject"  # a subscript or attribute of a PythonObject (`.copy()` too)
    if nxt == "(":
        if head.startswith("py_"):
            # a C-API helper: its declared return type (py_get / py_call / py_item return PythonObject; py_int_list
            # a List); not known: not typed
            return return_type(head, local, glob)
        if head[:1].isupper() or (head[:1] == "_" and head[1:2].isupper()):
            return head  # a constructor
        return return_type(head, local, glob)
    return None


def functions(lines):
    """[(name, first line index, end index exclusive, indent)] of every def, nested ones included."""
    out = []
    for i, line in enumerate(lines):
        m = DEF_RE.match(line)
        if not m:
            continue
        ind = len(m.group(1))
        j = i + 1
        # the signature may span lines: skip to the line that ends it
        depth = line.count("(") - line.count(")")
        while depth > 0 and j < len(lines):
            depth += lines[j].count("(") - lines[j].count(")")
            j += 1
        k = j
        while k < len(lines) and (not lines[k].strip() or indent(lines[k]) > ind or lines[k].lstrip().startswith("#")):
            k += 1
        out.append((m.group(2), i, k, ind, j))
    return out


def scan_file(path, text, local, glob, fields):
    """[(line number, function, kind, token, code)] of the findings, and the region count."""
    raw = text.splitlines()
    lines = [strip(l) for l in raw]
    defs = functions(lines)
    findings, regions = [], 0
    for i, line in enumerate(lines):
        m = WITH_RE.match(line)
        if not m:
            continue
        regions += 1
        ind = len(m.group(1))
        # the innermost function holding the region
        owner = None
        for d in defs:
            if d[1] < i < d[2] and (owner is None or d[1] > owner[1]):
                owner = d
        fname = owner[0] if owner else "<module>"
        names, types = set(), {}
        if owner:
            sig = " ".join(lines[owner[1]:owner[4]])
            names.update(PARAM_RE.findall(sig))
            for pname, ptype in PARAM_T_RE.findall(sig):
                if ptype != "PythonObject":
                    types[pname] = ptype
            # NEAREST DECLARATION WINS: a name re-declared in another branch or loop of the function is typed by
            # its last `var` before the region (else the first after it, a declaration inside the region).
            # Three passes so that names assigned from names settle.
            for _ in range(3):
                latest = {}
                for k in range(owner[4], owner[2]):
                    l = lines[k]
                    decl = None
                    t = VAR_T_RE.match(l)
                    if t:
                        decl = (t.group(1), t.group(2))
                    else:
                        a = VAR_ASSIGN_RE.match(l)
                        if a:
                            decl = (a.group(1), expr_type(a.group(2), names, types, local, glob))
                        else:
                            f = FOR_RE.match(l)
                            if f:
                                decl = (f.group(1), "PythonObject" if f.group(2) in names else None)
                    if decl is None:
                        continue
                    name, typ = decl
                    if k < i or name not in latest:
                        latest[name] = typ
                for name, typ in latest.items():
                    if typ == "PythonObject":
                        names.add(name)
                        types.pop(name, None)
                    else:
                        names.discard(name)
                        if typ:
                            types[name] = typ
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or indent(lines[j]) > ind):
            body = lines[j]
            if body.strip():
                hits = []
                for p in PY_API_RE.finditer(body):
                    hits.append(("py-api", p.group(0).strip()))
                for p in IDENT_RE.finditer(body):
                    tok = p.group(1)
                    rest = body[p.end():].lstrip()
                    if tok in names and not rest.startswith(":"):
                        hits.append(("py-name", tok))
                for p in FIELD_USE_RE.finditer(body):
                    if p.group(2) in fields.get(types.get(p.group(1), ""), ()):
                        hits.append(("py-field", p.group(1) + "." + p.group(2)))
                for p in CALL_RE.finditer(body):
                    if return_type(p.group(1), local, glob) == "PythonObject":
                        hits.append(("py-return", p.group(1)))
                seen = set()
                for kind, tok in hits:
                    if (kind, tok) in seen:
                        continue
                    seen.add((kind, tok))
                    findings.append((j + 1, fname, kind, tok, raw[j].strip()))
            j += 1
    return findings, regions


def load_allow(root, commit, path):
    text = ""
    if commit is None:
        p = Path(root) / path
        text = p.read_text(encoding="utf-8") if p.is_file() else ""
    else:
        r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root, capture_output=True, text=True)
        text = r.stdout if r.returncode == 0 else ""
    allow = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        allow.add(line.split()[0])
    return allow


def main(argv):
    args = argv[1:]
    root = Path(args[args.index("--root") + 1]) if "--root" in args else ROOT
    commit = args[args.index("--commit") + 1] if "--commit" in args else None
    allow_path = args[args.index("--allow") + 1] if "--allow" in args else ALLOW
    verbose = "--verbose" in args
    files = files_of(root, commit)
    per_file, glob, fields = collect(files)
    allow = load_allow(root, commit, allow_path)
    bad = allowed = regions = 0
    touched = set()
    for path, text in files:
        if "GILReleased" not in text:
            continue
        findings, n = scan_file(path, text, per_file[path], glob, fields)
        regions += n
        for line_no, fname, kind, tok, code in findings:
            key = f"{path}:{fname}:{tok}"
            if key in allow:
                allowed += 1
                if verbose:
                    print(f"GILREGION ALLOWED {path}:{line_no} {fname}: {kind} {tok}")
                continue
            bad += 1
            touched.add(path)
            print(f"GILREGION {path}:{line_no} {fname}: {kind} `{tok}` | {code[:160]}")
    where = commit[:12] if commit else str(root)
    print(f"GILREGION {'FAIL' if bad else 'PASS'} {regions} released regions in {len(files)} files of {where}: "
          f"{bad} findings in {len(touched)} files, {allowed} allowed ({allow_path})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
