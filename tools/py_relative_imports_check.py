"""Static gate for the Python package: every relative `from .mod import name` under python/mojolearn names a module
file (or package) that exists, and every name imported from a sibling .py module is defined or imported there.
No import is executed (no compiled binding is needed), so it runs anywhere in seconds:
`python3 tools/py_relative_imports_check.py` -> exit 1 and one line per broken import.
Each file is an independent unit and is parsed in a process pool sized by the host's cores."""
import ast, os, sys
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python", "mojolearn")

def names_of(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out, star = set(), False
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): out.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name == "*": star = True
                out.add((a.asname or a.name).split(".")[0])
    return out, star

def check(path):
    bad = []
    try: tree = ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError as e: return [f"{path}: syntax error {e}"]
    pkg = os.path.dirname(path)
    for n in ast.walk(tree):
        if not isinstance(n, ast.ImportFrom) or not n.level: continue
        base = pkg
        for _ in range(n.level - 1): base = os.path.dirname(base)
        target = os.path.join(base, *(n.module.split("."))) if n.module else base
        mod_py, mod_pkg = target + ".py", os.path.join(target, "__init__.py")
        if n.module and not (os.path.exists(mod_py) or os.path.exists(mod_pkg) or os.path.isdir(target)):
            if not any(f.startswith(os.path.basename(target) + ".") for f in os.listdir(os.path.dirname(target)) if not f.endswith(".py")):
                bad.append(f"{os.path.relpath(path, ROOT)}:{n.lineno}: no module {'.' * n.level}{n.module}")
            continue
        src = mod_py if os.path.exists(mod_py) else (mod_pkg if os.path.exists(mod_pkg) else None)
        if not src: continue
        defined, star = names_of(src)
        if star: continue
        for a in n.names:
            if a.name == "*" or a.name in defined: continue
            sub = os.path.join(os.path.dirname(src) if src.endswith("__init__.py") else target, a.name)
            if os.path.exists(sub + ".py") or os.path.isdir(sub): continue
            if src.endswith("__init__.py") and "__getattr__" in defined: continue
            bad.append(f"{os.path.relpath(path, ROOT)}:{n.lineno}: {'.' * n.level}{n.module or ''} has no name {a.name}")
    return bad

def main():
    files = [os.path.join(d, f) for d, _, fs in os.walk(ROOT) for f in fs if f.endswith(".py")]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        bad = [b for r in pool.map(check, files, chunksize=32) for b in r]
    for b in sorted(bad): print(b)
    print(f"py_relative_imports_check: {len(files)} files, {len(bad)} broken")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
