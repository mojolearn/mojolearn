#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The scikit-learn twin table: per public Mojolearn class that shares its name
with a scikit-learn estimator, the two constructors (parameter names, defaults,
accepted values), the public methods and the fitted attributes, side by side,
with every difference as a row and each difference's reason.

    python3 tools/api_twin_table.py                 # write docs/api/SKLEARN_TWINS.md + sklearn_twins.json
    python3 tools/api_twin_table.py --check         # exit 1 when a difference has no reason
    python3 tools/api_twin_table.py --write-snapshot  # (needs scikit-learn) refresh the snapshot

OUR side is read from the package SOURCE with `ast` (no import, no binary;
pattern: tools/device_class_table.py): the public names of
python/mojolearn/__init__.py, each class's `__init__` (through its bases,
resolved per module), the public methods in the class bodies, and the fitted
attributes listed in the class docstring's ``Attributes`` section (the native
PACK stores them; the docstring is their documented contract).

SCIKIT-LEARN's side comes from the installed scikit-learn by introspection when
it is importable, else from the checked-in snapshot docs/api/sklearn_snapshot.json
(`--write-snapshot` regenerates it; `--snapshot` forces it). Accepted values are
scikit-learn's `_parameter_constraints`; ours are the docstring's type line,
shown beside it (our values are checked by the compiled constructor).

Each difference is matched against docs/api/sklearn_twin_reasons.json, a list of
rules {kind, class, name, reason} (`class`/`name` are fnmatch globs; the first
matching rule wins). A difference no rule matches is UNCLASSIFIED and `--check`
fails. Testing tool only: one process, one pass over the files, nothing here is
product code, so there is nothing to split across cores.
"""
import ast
import fnmatch
import inspect
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
ROOT = os.path.join(REPO, "python", "mojolearn")
DOCS = os.path.join(REPO, "docs", "api")
SNAPSHOT = os.path.join(DOCS, "sklearn_snapshot.json")
REASONS = os.path.join(DOCS, "sklearn_twin_reasons.json")
OUT_MD = os.path.join(DOCS, "SKLEARN_TWINS.md")
OUT_JSON = os.path.join(DOCS, "sklearn_twins.json")

# Methods every scikit-learn estimator inherits for metadata routing and output
# configuration; they are framework plumbing, not the estimator's API, and are
# compared nowhere (Mojolearn has no metadata routing).
SKLEARN_PLUMBING = re.compile(r"^(set_\w+_request|get_metadata_routing|set_output|__sklearn\w+)$")
# Mojolearn's own process-level methods on every estimator (numeric mode, device
# reports); they are listed under "extra" with one shared reason.
KINDS = ("positional", "param_missing", "param_extra", "default", "method_missing", "method_extra",
         "attribute_missing", "attribute_extra")


# ---------------------------------------------------------------- docstrings
def doc_section(doc, title):
    """The `name : type` entries of one numpydoc section, as {name: type_text}."""
    if not doc:
        return {}
    lines = inspect.cleandoc(doc).splitlines()
    out = {}
    inside = False
    for i, line in enumerate(lines):
        if line.strip() == title and i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            inside = True
            continue
        if not inside:
            continue
        if set(line.strip()) == {"-"}:
            continue
        if line and not line.startswith(" ") and i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            break  # the next section's title
        if line and not line.startswith(" "):
            head, _, kind = line.partition(" : ")
            for name in head.split(","):
                name = name.strip().strip("`*")
                if re.fullmatch(r"\w+", name):
                    out[name] = kind.strip()
    return out


# ---------------------------------------------------------------- our side
class _Module:
    def __init__(self, name, tree):
        self.name = name
        self.classes = {}
        self.imports = {}  # local name -> (module, name)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
            elif isinstance(node, ast.ImportFrom) and node.level >= 1:
                mod = (node.module or "").split(".")[0] if node.level == 1 else None
                for a in node.names:
                    self.imports[a.asname or a.name] = (mod, a.name)


def _load_modules():
    mods = {}
    for name in sorted(os.listdir(ROOT)):
        if name.endswith(".py"):
            path = os.path.join(ROOT, name)
            mods[name[:-3]] = _Module(name[:-3], ast.parse(open(path, encoding="utf-8").read()))
    return mods


def _public_names(mods):
    """{public name: module or None}, from __init__'s _LAZY_NAMES, __all__ and imports."""
    init = ast.parse(open(os.path.join(ROOT, "__init__.py"), encoding="utf-8").read())
    where = {}
    for node in init.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "_LAZY_NAMES":
                for k, v in ast.literal_eval(node.value).items():
                    if v:
                        where[k] = v.lstrip(".").split(".")[0]
            elif node.targets[0].id == "__all__":
                for k in ast.literal_eval(node.value):
                    where.setdefault(k, None)
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for a in node.names:
                where.setdefault(a.asname or a.name, node.module.split(".")[0])
    return where


def _find_class(mods, module, name, depth=0):
    """(module name, ClassDef) for `name` as seen from `module`; follows re-exports."""
    if depth > 8:
        return None
    m = mods.get(module) if module else None
    if m is not None:
        if name in m.classes:
            return module, m.classes[name]
        if name in m.imports:
            src, real = m.imports[name]
            if src:
                return _find_class(mods, src, real, depth + 1)
    for mname, mm in mods.items():  # last resort: a unique definition anywhere
        if name in mm.classes and (m is None or mname != module):
            return mname, mm.classes[name]
    return None


def _ancestry(mods, module, node, seen=None):
    seen = [] if seen is None else seen
    if any(n is node for _, n in seen):
        return seen
    seen.append((module, node))
    for base in node.bases:
        bname = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", None)
        if not bname or bname == "object":
            continue
        found = _find_class(mods, module, bname)
        if found:
            _ancestry(mods, found[0], found[1], seen)
    return seen


def _src(node):
    """A default as the text `repr` would give for its value: literals and constant
    expressions (`float('nan')`, `2.0 ** -20`, tuples) are evaluated, names stay source."""
    def value(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Tuple):
            return tuple(value(e) for e in n.elts)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -value(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Pow, ast.Mult, ast.Div, ast.Add, ast.Sub)):
            a, b = value(n.left), value(n.right)
            return {ast.Pow: lambda: a ** b, ast.Mult: lambda: a * b, ast.Div: lambda: a / b,
                    ast.Add: lambda: a + b, ast.Sub: lambda: a - b}[type(n.op)]()
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "float"
                and len(n.args) == 1 and isinstance(n.args[0], ast.Constant)):
            return float(n.args[0].value)
        raise ValueError
    try:
        return repr(value(node))
    except Exception:
        return ast.unparse(node)


def _init_params(func):
    a = func.args
    out = {}
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(pos, defaults):
        if arg.arg != "self":
            out[arg.arg] = {"default": _src(d) if d is not None else "<required>", "positional": True}
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        out[arg.arg] = {"default": _src(d) if d is not None else "<required>"}
    if a.kwarg:
        out["**" + a.kwarg.arg] = {"default": "<any keyword>"}
    return out


def _is_method_value(value):
    return isinstance(value, (ast.Attribute, ast.Name, ast.Call, ast.Lambda, ast.Subscript))


def our_class(mods, module, name):
    found = _find_class(mods, module, name)
    if not found:
        return None
    line = _ancestry(mods, found[0], found[1])
    params, owner = None, None
    methods, props, conditional = set(), set(), {}
    doc_attrs, doc_params = None, {}
    for mod, node in line:
        doc = ast.get_docstring(node)
        if doc_attrs is None and doc and "Attributes" in doc:
            doc_attrs = doc_section(doc, "Attributes")
        for k, v in doc_section(doc, "Parameters").items():
            doc_params.setdefault(k, v)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "__init__" and params is None:
                    params, owner = _init_params(item), node.name
                elif item.name.endswith("_") and not item.name.startswith("_"):
                    props.add(item.name)  # a fitted attribute served by a property
                elif not item.name.startswith("_"):
                    methods.add(item.name)
                    for dec in item.decorator_list:  # @available_if(lambda self: hasattr(self, "transform"))
                        need = re.search(r"available_if\(lambda self: hasattr\(self, ['\"](\w+)['\"]\)\)",
                                         ast.unparse(dec))
                        if need:
                            conditional[item.name] = need.group(1)
            elif isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                t = item.targets[0].id
                if t == "__init__" and params is None:
                    params, owner = {"**<connected constructor>": {"default": ast.unparse(item.value)}}, node.name
                elif t.endswith("_") and not t.startswith("_"):
                    props.add(t)
                elif not t.startswith("_") and not t.isupper() and _is_method_value(item.value):
                    methods.add(t)
    methods -= {m for m, need in conditional.items() if need not in methods}
    for p, row in (params or {}).items():
        row["accepts"] = doc_params.get(p, "")
    return {"module": found[0], "init_owner": owner, "params": params or {},
            "methods": sorted(methods), "attributes": sorted(set(doc_attrs or {}) | props)}


# ---------------------------------------------------------------- scikit-learn's side
def _constraint_text(c):
    if isinstance(c, str):
        return c
    name = type(c).__name__
    if name == "Interval":
        lo = "-inf" if c.left is None else repr(c.left)
        hi = "inf" if c.right is None else repr(c.right)
        lb = "[" if c.closed in ("left", "both") else "("
        rb = "]" if c.closed in ("right", "both") else ")"
        return f"{getattr(c.type, '__name__', c.type)} in {lb}{lo}, {hi}{rb}"
    if name == "StrOptions" or name == "Options":
        return "{" + ", ".join(repr(o) for o in sorted(c.options, key=repr)) + "}"
    if name == "HasMethods":
        return "object with " + ", ".join(c.methods)
    if isinstance(c, type):
        return c.__name__
    if c is None:
        return "None"
    return name


def sklearn_side():
    import sklearn
    from sklearn.utils import all_estimators
    out = {}
    for name, cls in all_estimators():
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        params = {}
        cons = getattr(cls, "_parameter_constraints", {}) or {}
        for p in list(sig.parameters.values())[1:]:
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            d = p.default
            if d is inspect.Parameter.empty:
                d = "<required>"
            elif isinstance(d, float) or type(d).__name__ in ("float64", "float32"):
                d = repr(float(d))  # np.float64(x) is the number x
            elif inspect.isfunction(d) or inspect.isclass(d):
                d = d.__name__  # a callable default: its name, since Mojolearn takes names
            else:
                d = repr(d)
            params[p.name] = {"default": d, "positional": p.kind == p.POSITIONAL_OR_KEYWORD, "accepts": " | ".join(_constraint_text(c) for c in cons.get(p.name, []))}
        methods = sorted(m for m in dir(cls) if not m.startswith("_") and not SKLEARN_PLUMBING.match(m)
                         and callable(getattr(cls, m, None)) and m not in ("get_params", "set_params"))
        out[name] = {"public_path": _public_path(cls), "params": params, "methods": methods,
                     "attributes": sorted(doc_section(cls.__doc__, "Attributes"))}
    return {"sklearn_version": sklearn.__version__, "classes": out}


def _public_path(cls):
    parts = cls.__module__.split(".")
    keep = [p for p in parts if not p.startswith("_")]
    return ".".join(keep[:2]) + "." + cls.__name__


# ---------------------------------------------------------------- comparison
def _same_default(a, b):
    """Both sides are `repr` text; numbers compare by value (1 == 1.0, 1e-06 == 1e-6)."""
    if a == b:
        return True
    try:
        x, y = ast.literal_eval(a), ast.literal_eval(b)
    except Exception:
        return False
    if isinstance(x, bool) or isinstance(y, bool):
        return False
    return isinstance(x, (int, float)) and isinstance(y, (int, float)) and float(x) == float(y)


def differences(cls, ours, theirs):
    rows = []
    op, tp = ours["params"], theirs["params"]
    open_kwargs = any(p.startswith("**") for p in op)
    for p in tp:
        if p not in op:
            rows.append(("param_missing", p, "via **kwargs (unchecked)" if open_kwargs else "", tp[p]["default"]))
        elif not _same_default(op[p]["default"], tp[p]["default"]):
            rows.append(("default", p, op[p]["default"], tp[p]["default"]))
    for p in op:
        if p not in tp:
            rows.append(("param_extra", p, op[p]["default"], ""))
    ours_pos = [p for p in op if op[p].get("positional")]
    their_pos = [p for p in tp if tp[p].get("positional")]
    # A call scikit-learn accepts passes only its own positional parameters, in its
    # order: they must be the first positional parameters here too.
    if ours_pos[:len(their_pos)] != their_pos:
        rows.append(("positional", "(" + ", ".join(their_pos) + ")", ", ".join(ours_pos) or "keyword-only",
                     ", ".join(their_pos) or "keyword-only"))
    om, tm = set(ours["methods"]), set(theirs["methods"])
    for m in sorted(tm - om):
        rows.append(("method_missing", m, "", "yes"))
    for m in sorted(om - tm - {"get_params", "set_params"}):
        rows.append(("method_extra", m, "yes", ""))
    oa, ta = set(ours["attributes"]), set(theirs["attributes"])
    for a in sorted(ta - oa):
        rows.append(("attribute_missing", a, "", "yes"))
    for a in sorted(oa - ta):
        rows.append(("attribute_extra", a, "yes", ""))
    return rows


def classify(rules, cls, kind, name, ours, theirs):
    """The reason of the first rule that matches; `class`, `name`, `ours` and
    `sklearn` are fnmatch globs (a list is any of them), `kind` a kind or a list."""
    def hit(pattern, value):
        pats = pattern if isinstance(pattern, list) else [pattern]
        return any(fnmatch.fnmatchcase(str(value), p) for p in pats)
    for r in rules:
        kinds = r["kind"] if isinstance(r["kind"], list) else [r["kind"]]
        if (kind in kinds and hit(r.get("class", "*"), cls) and hit(r.get("name", "*"), name)
                and hit(r.get("ours", "*"), ours) and hit(r.get("sklearn", "*"), theirs)):
            return r["reason"]
    return None


# ---------------------------------------------------------------- output
def _cell(v):
    return str(v).replace("|", "\\|").replace("\n", " ")


def write(table, sk_version, source):
    os.makedirs(DOCS, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"sklearn_version": sk_version, "sklearn_source": source, "classes": table}, f,
                  indent=1, default=str)
        f.write("\n")
    total = sum(len(t["differences"]) for t in table.values())
    open_rows = sum(1 for t in table.values() for d in t["differences"] if not d["reason"])
    md = ["# Mojolearn and its scikit-learn twins", "",
          "Generated by `python3 tools/api_twin_table.py`; do not edit by hand. Reasons live in",
          "`docs/api/sklearn_twin_reasons.json`. Machine-readable form: `docs/api/sklearn_twins.json`.", "",
          f"scikit-learn {sk_version} ({source}). Classes with a twin: {len(table)}. "
          f"Differences: {total}. Without a reason: {open_rows}.", ""]
    for cls in sorted(table):
        t = table[cls]
        md += [f"## {cls}", "", f"`mojolearn.{cls}` (`{t['module']}.py`) and `{t['twin']}`.", ""]
        md += ["| parameter | Mojolearn default | scikit-learn default | Mojolearn accepts | scikit-learn accepts |",
               "|---|---|---|---|---|"]
        names = list(t["sklearn"]["params"]) + [p for p in t["ours"]["params"] if p not in t["sklearn"]["params"]]
        for p in names:
            o = t["ours"]["params"].get(p)
            s = t["sklearn"]["params"].get(p)
            md.append("| `{}` | {} | {} | {} | {} |".format(
                p, _cell(o["default"]) if o else "absent", _cell(s["default"]) if s else "absent",
                _cell(o.get("accepts", "")) if o else "", _cell(s["accepts"]) if s else ""))
        md += ["", "Methods: " + ", ".join(f"`{m}`" for m in t["ours"]["methods"]) + ".",
               "Fitted attributes: " + (", ".join(f"`{a}`" for a in t["ours"]["attributes"]) or "none documented") + ".", ""]
        if t["differences"]:
            md += ["| difference | name | Mojolearn | scikit-learn | reason |", "|---|---|---|---|---|"]
            for d in t["differences"]:
                md.append("| {} | `{}` | {} | {} | {} |".format(d["kind"], d["name"], _cell(d["ours"]),
                                                              _cell(d["sklearn"]), _cell(d["reason"] or "UNCLASSIFIED")))
            md.append("")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return total, open_rows


def main(argv):
    if "--write-snapshot" in argv:
        snap = sklearn_side()
        os.makedirs(DOCS, exist_ok=True)
        with open(SNAPSHOT, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=1)
            f.write("\n")
        print("snapshot:", SNAPSHOT, "scikit-learn", snap["sklearn_version"], "classes", len(snap["classes"]))
        return 0
    source = "snapshot"
    if "--snapshot" not in argv:
        try:
            sk = sklearn_side()
            source = "installed"
        except ImportError:
            sk = None
    if source == "snapshot" or sk is None:
        sk = json.load(open(SNAPSHOT, encoding="utf-8"))
        source = "snapshot"
    rules = json.load(open(REASONS, encoding="utf-8"))["rules"] if os.path.exists(REASONS) else []
    mods = _load_modules()
    table = {}
    missing_class = []
    for name, module in sorted(_public_names(mods).items()):
        if name not in sk["classes"]:
            continue
        ours = our_class(mods, module, name)
        if ours is None:
            missing_class.append(name)
            continue
        theirs = sk["classes"][name]
        rows = [dict(kind=k, name=n, ours=o, sklearn=s, reason=classify(rules, name, k, n, o, s))
                for k, n, o, s in differences(name, ours, theirs)]
        table[name] = {"module": ours["module"], "twin": theirs["public_path"], "ours": ours, "sklearn": theirs,
                       "differences": rows}
    total, open_rows = write(table, sk["sklearn_version"], source)
    by_kind = {}
    for t in table.values():
        for d in t["differences"]:
            if not d["reason"]:
                by_kind[d["kind"]] = by_kind.get(d["kind"], 0) + 1
    print(f"twins: {len(table)}  differences: {total}  unclassified: {open_rows}  "
          f"scikit-learn {sk['sklearn_version']} ({source})")
    if by_kind:
        print("unclassified by kind:", json.dumps(by_kind, sort_keys=True))
    if missing_class:
        print("public names whose class was not found in the source:", ", ".join(missing_class))
    if "--check" in argv and (open_rows or missing_class):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
