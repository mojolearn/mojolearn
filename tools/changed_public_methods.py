#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""List the public Python methods whose source changed between two revisions.

Release gate helper (docs/plans/TEST_COVERAGE_20261021.md): every public
endpoint that is new or changed since the last published release needs tests.

For each file under python/mojolearn/ (tests excluded) that differs between
BASE and HEAD, both versions are parsed and every def is keyed by its
qualified name (Class.method, outer.inner). A def is reported when it is new
or its AST differs (docstring-only edits included as `doc`). A def is public
when no part of its qualified name starts with `_` (dunder `__call__`,
`__init__` count as public) and the module is reachable from the package
export walk or has no leading underscore.

  python3 tools/changed_public_methods.py [BASE] [HEAD]     # TSV on stdout
  python3 tools/changed_public_methods.py --all [BASE]      # private defs too
  python3 tools/changed_public_methods.py --endpoints [BASE] # public endpoints only, with the
      changed private helpers their call path reaches (walk of tools/public_runtime_inventory.py)

Columns: status(new|changed|doc|removed)  module  qualname  line  public
Static, Python only: nothing is imported.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PUBLIC_DUNDERS = {"__init__", "__call__", "__len__", "__getitem__", "__iter__", "__enter__", "__exit__"}


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def _source(rev, path):
    try:
        return _git("show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return None


def _strip_doc(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def defs(text):
    """{qualname: (lineno, full dump, dump without docstring)}"""
    out = {}
    if text is None:
        return out
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out

    def walk(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = f"{prefix}{node.name}"
                if not isinstance(node, ast.ClassDef):
                    inner = _strip_doc(node)
                    sig = ast.dump(node.args) + ast.dump(ast.List(elts=node.decorator_list))
                    out[qual] = (node.lineno, ast.dump(node), sig + "".join(ast.dump(s) for s in inner))
                walk(node.body, qual + ".")
    walk(tree.body, "")
    return out


def is_public(module, qual):
    parts = qual.split(".")
    for p in parts:
        if p.startswith("_") and p not in PUBLIC_DUNDERS:
            return False
    return True


def changed_defs(base, head):
    """[(status, module, qualname, line, public)] for every def that differs."""
    files = [f for f in _git("diff", "--name-only", f"{base}..{head}", "--", "python/mojolearn/",
                             ":!python/mojolearn/tests").split() if f.endswith(".py")]
    out = []
    for path in files:
        old, new = defs(_source(base, path)), defs(_source(head, path))
        module = path[len("python/"):-3].replace("/", ".").removesuffix(".__init__")
        for qual in sorted(set(old) | set(new)):
            pub = is_public(module, qual)
            if qual not in old:
                status, line = "new", new[qual][0]
            elif qual not in new:
                status, line = "removed", old[qual][0]
            elif old[qual][1] == new[qual][1]:
                continue
            elif old[qual][2] == new[qual][2]:
                status, line = "doc", new[qual][0]
            else:
                status, line = "changed", new[qual][0]
            out.append((status, module, qual, line, pub))
    return out


def public_origins():
    """{'module.Name'}: the export walk of the runtime inventory plus every `__all__` name of a
    public (no leading underscore) module of the package."""
    sys.path.insert(0, str(REPO / "tools"))
    import public_runtime_inventory as inv
    origins = {origin for _c, origin, _a in inv.public_names(REPO)}
    pkg = REPO / "python" / "mojolearn"
    for path in pkg.rglob("*.py"):
        rel = path.relative_to(pkg.parent)
        if "tests" in rel.parts or any(p.startswith("_") and p != "__init__.py" for p in rel.parts):
            continue
        module = ".".join(rel.with_suffix("").parts).removesuffix(".__init__")
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
                try:
                    origins.update(f"{module}.{n}" for n in ast.literal_eval(node.value))
                except ValueError:
                    pass
    return origins, inv


def endpoints(base, head):
    rows = changed_defs(base, head)
    origins, inv = public_origins()
    changed = {(m, q) for st, m, q, _l, _p in rows if st != "doc"}
    print("kind\tstatus\tendpoint\tline\tchanged_helpers")
    seen = set()
    for st, m, q, line, pub in rows:
        top, _, rest = q.partition(".")
        if f"{m}.{top}" in origins and is_public(m, q):
            seen.add((m, q))
            print(f"direct\t{st}\t{m}.{q}\t{line}\t")
    # Public operation methods whose walked call path reaches a changed def.
    index = inv.Index(REPO)
    cache = {}
    for canonical, origin, _aliases in inv.public_names(REPO):
        module, _, name = origin.rpartition(".")
        target = index.resolve_name(module, name) if module else None
        if target is None or target[0] not in ("func", "class"):
            continue
        starts = []
        if target[0] == "func":
            starts.append((target[1], None, target[1].qual))
        else:
            for op in inv.OPS + ("__init__",):
                found = index.lookup(target[1], op)
                if found and found[0] == "func":
                    starts.append((found[1], target[1], f"{target[1].qual}.{op}"))
        for start, klass, label in starts:
            if (start.mod.name, start.qual) in seen:
                continue
            reached, stack = set(), [(start, klass)]
            while stack:
                item, k = stack.pop()
                key = (item.key, k.key if k else None)
                if key in reached:
                    continue
                reached.add(key)
                if key not in cache:
                    cache[key] = inv.analyze(index, item, k)
                stack.extend(cache[key].callees)
            hits = sorted({f"{fk[0]}.{fk[1]}" for fk, _ in reached
                           if fk in changed and fk != (start.mod.name, start.qual)})
            if hits:
                print(f"path\tchanged\t{origin.rpartition('.')[0]}.{label}\t{start.node.lineno}\t{','.join(hits)}")
    return 0


def main(argv):
    show_all = "--all" in argv
    argv = [a for a in argv if a != "--all"]
    if "--endpoints" in argv:
        argv = [a for a in argv if a != "--endpoints"]
        return endpoints(argv[0] if argv else "fc8d425fb", argv[1] if len(argv) > 1 else "HEAD")
    base = argv[0] if argv else "fc8d425fb"
    head = argv[1] if len(argv) > 1 else "HEAD"
    print("status\tmodule\tqualname\tline\tpublic")
    for status, module, qual, line, pub in changed_defs(base, head):
        if not pub and not show_all:
            continue
        print(f"{status}\t{module}\t{qual}\t{line}\t{int(pub)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
