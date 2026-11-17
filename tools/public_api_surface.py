"""Print the public Python surface of a python/mojolearn tree as sorted lines, for diffing two trees:
`python3 tools/public_api_surface.py <tree>/python/mojolearn > surface.txt`.
Lines: `module:<public module>`, `export:<module>.<name>` (from __all__ or public top-level names), and
`method:<Class>.<name>` for every public class anywhere in the package (tests excluded), with attributes
inherited from package classes resolved by class name. Static (ast only): nothing is imported or run.
Files are independent units parsed in a process pool sized by the host's cores."""
import ast, os, sys
from concurrent.futures import ProcessPoolExecutor

def scan(args):
    root, path = args
    rel = os.path.relpath(path, root)[:-3].replace(os.sep, ".")
    try: tree = ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError: return rel, None, [], {}
    allv, tops, classes = None, [], {}
    # `from m import X as _X`: a base spelled `_X` is the class `X` (aliasing a base does not change the class)
    alias = {a.asname: a.name for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names if a.asname}
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            try: allv = [str(x) for x in ast.literal_eval(n.value)]
            except Exception: pass
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)): tops.append(n.name)
        elif isinstance(n, ast.Assign):
            tops += [t.id for t in n.targets if isinstance(t, ast.Name)]
        elif isinstance(n, ast.ImportFrom):
            tops += [(a.asname or a.name) for a in n.names if a.name != "*"]
        if isinstance(n, ast.ClassDef):
            attrs = set()
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): attrs.add(m.name)
                elif isinstance(m, ast.Assign): attrs |= {t.id for t in m.targets if isinstance(t, ast.Name)}
                elif isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name): attrs.add(m.target.id)
            bases = [alias.get(b.id, b.id) if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else "")
                     for b in n.bases]
            classes[n.name] = (attrs, bases)
    return rel, allv, tops, classes

def main(root):
    files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs
             if f.endswith(".py") and os.sep + "tests" not in d and not f.startswith("test_")]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        res = list(pool.map(scan, [(root, f) for f in files], chunksize=32))
    allc = {}
    for rel, _, _, classes in res:
        for k, v in classes.items(): allc.setdefault(k, v)
    def attrs(name, seen=()):
        if name not in allc or name in seen: return set()
        a, bases = allc[name]; out = set(a)
        for b in bases: out |= attrs(b, seen + (name,))
        return out
    lines = set()
    for rel, allv, tops, classes in res:
        parts = rel.split(".")
        public_mod = not any(p.startswith("_") and p != "__init__" for p in parts)
        if public_mod:
            mod = ".".join(p for p in parts if p != "__init__") or "mojolearn"
            lines.add(f"module:{mod}")
            for n in (allv if allv is not None else [t for t in tops if not t.startswith("_")]):
                lines.add(f"export:{mod}.{n}")
        for c in classes:
            if c.startswith("_"): continue
            for a in attrs(c):
                if not a.startswith("_"): lines.add(f"method:{c}.{a}")
    print("\n".join(sorted(lines)))

if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]))
