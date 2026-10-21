#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The measure of the public runtime migration (docs/plans/PUBLIC_RUNTIME_MIGRATION.md).

The contract: a public method is argument checks, a choice of binding, ONE
native call that prepares, executes and produces the results, and a return.
This tool reads the Python call path of every public operation method and
says how far each one is from that.

For every public name `mojolearn` exports (the export walk of
packaging/native_only/inventory.py: the literal `__all__` of the package and of
its `_expansion_*` doors, plus the census rows of
packaging/native_only/manifest.json, deduplicated by symbol origin) and each
of its operation methods (OPS below), it statically walks the method body and
every mojolearn-internal helper the body calls, transitively, inside
python/mojolearn/. The walk stops at a native binding call: an entry called on
a binding object (`_backend.binding(...).entry(...)`, `_binding()._module.entry`,
`self._ext.entry(...)` where `_ext` holds a binding, a local `fn = b.entry`),
or an unresolved `_native_*(...)` call. Functions that return a binding
(`binding`, `*_binding`, `_bind`, and any helper whose return is a binding,
found by a fixpoint) are binding acquisition: counted nowhere, never walked.
Calls into the conversion layer (SHELL_MODULES) are counted as shell calls
and not walked; an allocation through it still counts as an allocation.

Per method it counts native entry call sites and distinct entries, Python
statements on the path, loops (untagged ones separately: a `# driver-loop:`
or `# glue` comment on the loop line or the line above tags it), branches that
choose a route (an `if` whose body is not only a raise and whose arms make a
call), NumPy calls (and the ones that are not conversion), allocations
(np.empty/zeros/ones/full and their *_like, `_buffer.empty/zeros/full`,
`*.alloc(...)`, `allocate_state`, `Array(...)`), helpers walked and calls it
could not resolve. The verdict:

  native        the method body makes the one native call itself: one native
                call site, no Python helper walked, no loop, route branch,
                allocation or NumPy work (a no-op such as a stateless `fit`
                that returns self also counts)
  python        no native call reached, or NumPy work, or an untagged loop
  orchestrated  everything else (several native calls, a route choice, an
                allocation, or a tagged driver loop)
  unresolved    the public name does not resolve to a def statically

Static analysis, Python only: nothing is imported or built.

  python3 tools/public_runtime_inventory.py                 # TSV on stdout, summary on stderr
  python3 tools/public_runtime_inventory.py --summary       # per-family numbers only
  python3 tools/public_runtime_inventory.py --baseline      # rewrite tools/hooks/public_runtime_baseline.tsv
  python3 tools/public_runtime_inventory.py --check         # fail if the non-native count grows
"""
from __future__ import annotations

import argparse
import ast
import builtins
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "tools/hooks/public_runtime_baseline.tsv"

OPS = ("fit", "partial_fit", "predict", "predict_proba", "decision_function", "transform",
       "fit_transform", "fit_predict", "inverse_transform", "score", "sample", "generate", "step",
       "forward", "train_step", "save", "load", "export", "__call__")
FUNCTION_OP = "call"   # the method column for a public function

VERDICTS = ("native", "orchestrated", "python", "unresolved")
RANK = {v: i for i, v in enumerate(VERDICTS)}

# The conversion and argument-check layer: calls into these are shell calls,
# not walked. Everything else in the package is walked.
SHELL_MODULES = frozenset({"_buffer", "_array", "_arrays", "_cast_in", "_bufcheck", "_sk_validate",
                           "_backend", "_optional_numpy", "_lazy_out", "_version", "exceptions"})

GETTER_NAMES = frozenset({"binding", "_bind", "load_host_module", "load_door_host", "load_set",
                          "_load_member", "_exec_binding"})
NUMPY_NAMES = frozenset({"np", "numpy", "_np"})
NUMPY_CONVERSION = frozenset({
    "asarray", "ascontiguousarray", "asfortranarray", "asanyarray", "array", "require", "dtype",
    "frombuffer", "isscalar", "issubdtype", "result_type", "can_cast", "iinfo", "finfo", "ndim",
    "shape", "size", "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8",
    "uint16", "uint32", "uint64", "bool_", "intp", "promote_types", "atleast_1d", "atleast_2d"})
ALLOC_NAMES = frozenset({"empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like",
                         "full_like", "alloc", "allocate", "allocate_state"})
ALLOC_CLASSES = frozenset({"Array"})
ALLOC_PATTERN = re.compile(r"^_?(empty|zeros|ones|full)(_like|_out|_array|_buffer)?$|alloc")
# Methods of plain containers and strings: not counted as unresolved.
PLAIN_METHODS = frozenset({
    "append", "extend", "insert", "pop", "get", "items", "keys", "values", "update", "setdefault",
    "copy", "join", "split", "strip", "lower", "upper", "startswith", "endswith", "format",
    "replace", "add", "discard", "remove", "clear", "index", "count", "encode", "decode",
    "rstrip", "lstrip", "partition", "rpartition", "sort", "reverse", "warn"})
BUILTINS = frozenset(dir(builtins))
NATIVE_MODULE = re.compile(r"^_mojolearn\w*$")
CONNECT_CALL = re.compile(r"^_?connect|native")
# Instance attributes the compiled runtime sets on the object itself.
NATIVE_ATTRS = frozenset({"_runtime"})
NATIVE_CALL = re.compile(r"^_native|_native$")
TAG = re.compile(r"#\s*(driver-loop:|glue)")


def _load_inventory():
    path = REPO / "packaging/native_only/inventory.py"
    spec = importlib.util.spec_from_file_location("native_only_inventory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- index

class Mod:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.tree = ast.parse(path.read_text(), filename=str(path))
        self.lines = path.read_text().splitlines()
        self.is_pkg = path.name == "__init__.py"
        self.short = name.rsplit(".", 1)[-1]
        self.defs, self.classes, self.imports, self.aliases, self.lazy = {}, {}, {}, {}, {}
        self.np_names = set()
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.aliases[target.id] = node.value
                        if target.id in ("_LAZY_EXPORTS", "_CPU_INFERENCE", "_DEPRECATED_ALIASES"):
                            try:
                                value = ast.literal_eval(node.value)
                            except (ValueError, TypeError):
                                continue
                            if isinstance(value, dict):
                                self.lazy.update({k: (target.id, v) for k, v in value.items()})
        _collect_imports(self, self.tree.body, self.imports, self.np_names, top=True)
        # `from . import _mojolearn_solver`: a compiled extension module, a binding
        self.binding_globals = {local for local, entry in self.imports.items()
                                if NATIVE_MODULE.match((entry[2] if entry[0] == "from" else entry[1]).rsplit(".", 1)[-1])}
        # `_NATIVE, _CONTEXT = _connect_causal_native()`: what a module-level connect returns
        for node in self.tree.body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and CONNECT_CALL.search(_final(node.value.func) or "")):
                for target in node.targets:
                    elts = target.elts if isinstance(target, ast.Tuple) else [target]
                    self.binding_globals.update(t.id for t in elts if isinstance(t, ast.Name))

    def line(self, lineno):
        return self.lines[lineno - 1] if 0 < lineno <= len(self.lines) else ""


def _import_base(mod, node):
    source = node.module or ""
    if node.level:
        package = mod.name if mod.is_pkg else mod.name.rpartition(".")[0]
        head = package.split(".")
        head = head[:len(head) - node.level + 1]
        source = ".".join(head + ([source] if source else []))
    return source


def _collect_imports(mod, body, imports, np_names, top=False):
    """Import aliases of statements in `body` (only top level when top)."""
    nodes = body if top else [n for s in body for n in ast.walk(s)]
    if top:
        # module-level imports can sit inside try/if blocks
        nodes = [n for s in body for n in ast.walk(s)
                 if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "numpy":
                    np_names.add(alias.asname or "numpy")
                else:
                    imports[alias.asname or alias.name.split(".")[0]] = (
                        "mod", alias.name if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            base = _import_base(mod, node)
            for alias in node.names:
                local = alias.asname or alias.name
                if base == "numpy" or base.startswith("numpy."):
                    np_names.add(local)
                else:
                    imports[local] = ("from", base, alias.name)
        elif (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
              and _final(node.value.func) == "require_numpy"):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    np_names.add(target.id)


def _relative(modname, is_pkg, target):
    if not target.startswith("."):
        return target
    package = modname if is_pkg else modname.rpartition(".")[0]
    level = len(target) - len(target.lstrip("."))
    head = package.split(".")
    head = head[:len(head) - level + 1]
    rest = target.lstrip(".")
    return ".".join(head + ([rest] if rest else []))


def _final(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _root_name(expr):
    while isinstance(expr, (ast.Attribute, ast.Subscript, ast.Call)):
        expr = expr.func if isinstance(expr, ast.Call) else expr.value
    return expr.id if isinstance(expr, ast.Name) else None


class Func:
    """A function definition with its module, defining class and scope parent."""

    def __init__(self, mod, node, cls=None, parent=None):
        self.mod, self.node, self.cls, self.parent = mod, node, cls, parent
        prefix = (parent.qual + ".<locals>.") if parent else (cls.qual + ".") if cls else ""
        self.qual = prefix + node.name
        self.key = (mod.name, self.qual)
        self.local_defs = {}
        self.imports, self.np_names = {}, set()
        for stmt in _own_nodes(node):
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.local_defs[stmt.name] = stmt
        _collect_imports(mod, [s for s in _own_nodes(node) if isinstance(s, (ast.Import, ast.ImportFrom, ast.Assign))],
                         self.imports, self.np_names, top=True)


class Cls:
    def __init__(self, mod, node):
        self.mod, self.node, self.name = mod, node, node.name
        self.qual = node.name
        self.key = (mod.name, node.name)


def _own_nodes(func_node):
    """Every node of a function body, not descending into nested defs/classes/lambdas' bodies."""
    out = []
    stack = list(reversed(func_node.body))
    while stack:
        node = stack.pop()
        out.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # its decorators and defaults are rare enough to ignore
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return out


class Index:
    def __init__(self, root, package="mojolearn"):
        self.root = root
        self.package = package
        self.base = root / "python" / package
        self.mods = {}
        self.funcs = {}
        self.classes = {}
        self.getters = set()
        self.entry_getters = set()
        self.returns_class = {}
        self.returns_tuple = {}
        self.returns_module = {}
        self.binding_params = {}
        self.binding_attrs = set(NATIVE_ATTRS)
        self._mro = {}
        self._prepass()

    # modules
    def module_file(self, dotted):
        if dotted != self.package and not dotted.startswith(self.package + "."):
            return None
        rel = dotted[len(self.package):].lstrip(".")
        path = self.base.joinpath(*rel.split(".")) if rel else self.base
        if path.with_suffix(".py").is_file():
            return path.with_suffix(".py")
        if (path / "__init__.py").is_file():
            return path / "__init__.py"
        return None

    def mod(self, dotted):
        if dotted in self.mods:
            return self.mods[dotted]
        path = self.module_file(dotted)
        mod = None
        if path is not None:
            try:
                mod = Mod(dotted, path)
            except SyntaxError:
                mod = None
        self.mods[dotted] = mod
        return mod

    def func(self, mod, node, cls=None, parent=None):
        prefix = (parent.qual + ".<locals>.") if parent else (cls.qual + ".") if cls else ""
        key = (mod.name, prefix + node.name)
        if key not in self.funcs:
            self.funcs[key] = Func(mod, node, cls, parent)
        return self.funcs[key]

    def cls(self, mod, node):
        key = (mod.name, node.name)
        if key not in self.classes:
            self.classes[key] = Cls(mod, node)
        return self.classes[key]

    # names
    def resolve_name(self, modname, name, depth=0):
        """('func', Func) | ('class', Cls) | ('module', dotted) | None."""
        if depth > 25:
            return None
        mod = self.mod(modname)
        if mod is None:
            return None
        if name in mod.defs:
            return ("func", self.func(mod, mod.defs[name]))
        if name in mod.classes:
            return ("class", self.cls(mod, mod.classes[name]))
        if name in mod.imports:
            entry = mod.imports[name]
            return self._resolve_import(entry, depth)
        if name in mod.aliases:
            value = mod.aliases[name]
            if isinstance(value, (ast.Name, ast.Attribute)):
                return self.resolve_expr(value, modname, None, depth + 1)
            if isinstance(value, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.BinOp,
                                  ast.UnaryOp, ast.JoinedStr)):
                return ("const", name)
        if name in mod.lazy:
            kind, target = mod.lazy[name]
            if kind == "_LAZY_EXPORTS" and isinstance(target, (tuple, list)) and len(target) == 2:
                return self.resolve_name(self.package + "." + target[0].lstrip("."), target[1], depth + 1)
            if kind == "_CPU_INFERENCE":
                if target is None:
                    return ("module", modname + "." + name)
                return self.resolve_name(_relative(modname, mod.is_pkg, target), name, depth + 1)
            if kind == "_DEPRECATED_ALIASES" and isinstance(target, str):
                return self.resolve_name(modname, target, depth + 1)
        if mod.is_pkg and self.module_file(modname + "." + name):
            return ("module", modname + "." + name)
        getter = mod.defs.get("__getattr__")
        if getter is not None:  # PEP 562: `from .x import name` inside the module __getattr__
            for node in ast.walk(getter):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if (alias.asname or alias.name) == name:
                            return self._resolve_import(("from", _import_base(mod, node), alias.name), depth)
        return None

    def _resolve_import(self, entry, depth):
        if entry[0] == "mod":
            return ("module", entry[1]) if self.module_file(entry[1]) else None
        _, base, name = entry
        if self.module_file(base + "." + name):
            return ("module", base + "." + name)
        if self.module_file(base):
            return self.resolve_name(base, name, depth + 1)
        return None

    def resolve_expr(self, expr, modname, func, depth=0):
        if depth > 25:
            return None
        if isinstance(expr, ast.Name):
            scope = func
            while scope is not None:
                if expr.id in scope.local_defs:
                    return ("func", self.func(scope.mod, scope.local_defs[expr.id], parent=scope))
                if expr.id in scope.imports:
                    return self._resolve_import(scope.imports[expr.id], depth)
                scope = scope.parent
            return self.resolve_name(modname, expr.id, depth + 1)
        if isinstance(expr, ast.Attribute):
            base = self.resolve_expr(expr.value, modname, func, depth + 1)
            if base is None:
                return None
            if base[0] == "module":
                return self.resolve_name(base[1], expr.attr, depth + 1)
            if base[0] == "class":
                return self.lookup(base[1], expr.attr)
        return None

    # classes
    def bases(self, cls):
        out = []
        for base in cls.node.bases:
            target = self.resolve_expr(base, cls.mod.name, None)
            if target and target[0] == "class":
                out.append(target[1])
        return out

    def mro(self, cls):
        if cls.key in self._mro:
            return self._mro[cls.key]
        self._mro[cls.key] = [cls]  # cycle guard
        order = [cls]
        for base in self.bases(cls):
            for item in self.mro(base):
                if item not in order:
                    order.append(item)
        self._mro[cls.key] = order
        return order

    def lookup(self, cls, attr, after=None):
        """Find `attr` on the class MRO (strictly after `after` when given)."""
        chain = self.mro(cls)
        if after is not None and after in chain:
            chain = chain[chain.index(after) + 1:]
        for klass in chain:
            for node in klass.node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == attr:
                    return ("func", self.func(klass.mod, node, cls=klass))
                if isinstance(node, ast.ClassDef) and node.name == attr:
                    return ("class", self.cls(klass.mod, node))
                if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == attr for t in node.targets):
                    value = node.value
                    if isinstance(value, ast.Name):
                        inner = self.lookup(klass, value.id) if value.id != attr else None
                        if inner:
                            return inner
                    if isinstance(value, (ast.Name, ast.Attribute)):
                        return self.resolve_expr(value, klass.mod.name, None)
                    return None
        return None

    def _flow(self, callee, call, flows):
        """A binding passed as an argument makes that parameter a binding in the callee."""
        positional, names = _params(callee.node)
        bound = isinstance(call.func, ast.Attribute) and positional[:1] in (["self"], ["cls"])
        offset = 1 if bound else 0
        for where, _ in flows:
            if isinstance(where, int):
                if where + offset < len(positional):
                    self.binding_params.setdefault(callee.key, set()).add(positional[where + offset])
            elif where in names:
                self.binding_params.setdefault(callee.key, set()).add(where)

    # pre-pass: which functions return a binding, which self attributes hold one
    def _prepass(self):
        for path in sorted(self.base.rglob("*.py")):
            rel = path.relative_to(self.base)
            if "tests" in rel.parts or rel.parts[0] in ("verify_reference", "reference_cards"):
                continue
            parts = list(rel.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            self.mod(".".join([self.package, *parts]))
        everything = []
        for mod in [m for m in self.mods.values() if m is not None]:
            for node in mod.tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    everything.append(self.func(mod, node))
                elif isinstance(node, ast.ClassDef):
                    klass = self.cls(mod, node)
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            everything.append(self.func(mod, sub, cls=klass))
        for _ in range(6):
            before = (len(self.getters), len(self.entry_getters), len(self.binding_attrs), len(self.returns_class),
                      len(self.returns_tuple), len(self.returns_module),
                      sum(map(len, self.binding_params.values())))
            for func in everything:
                scope = Scope(self, func, func.cls)
                for node in _own_nodes(func.node):
                    if isinstance(node, ast.Return) and node.value is not None:
                        made = scope.made_class(node.value)
                        if made is not None and func.key not in self.returns_class:
                            self.returns_class[func.key] = made
                        if isinstance(node.value, (ast.Name, ast.Attribute)):
                            made = self.resolve_expr(node.value, func.mod.name, func)
                            if made and made[0] == "module":
                                self.returns_module[func.key] = made[1]
                        if isinstance(node.value, ast.Tuple) and func.key not in self.returns_tuple:
                            made = [scope.made_class(elt) for elt in node.value.elts]
                            if any(made):
                                self.returns_tuple[func.key] = made
                        if scope.is_binding(node.value):
                            self.getters.add(func.key)
                        elif scope.entry_of(node.value) or (isinstance(node.value, ast.Name)
                                                            and node.value.id in scope.entry_names):
                            self.entry_getters.add(func.key)
                    if isinstance(node, ast.Call) and not isinstance(node.func, ast.Call):
                        flows = [(i, a) for i, a in enumerate(node.args) if scope.is_binding(a)]
                        flows += [(k.arg, k.value) for k in node.keywords if k.arg and scope.is_binding(k.value)]
                        if flows:
                            target = scope.resolve_call(node.func)
                            if target and target[0] == "func":
                                self._flow(target[1], node, flows)
                    if isinstance(node, ast.Assign) and scope.is_binding(node.value):
                        for target in node.targets:
                            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                                    and target.value.id in ("self", "cls")):
                                self.binding_attrs.add(target.attr)
            if (len(self.getters), len(self.entry_getters), len(self.binding_attrs),
                    len(self.returns_class), len(self.returns_tuple), len(self.returns_module),
                    sum(map(len, self.binding_params.values()))) == before:
                break


def _params(func_node):
    a = func_node.args
    return [p.arg for p in (*a.posonlyargs, *a.args)], {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}


class Scope:
    """Name facts of one function as reached from one root class."""

    def __init__(self, index, func, root_cls):
        self.index, self.func, self.root = index, func, root_cls
        self.modname = func.mod.name
        self.np_names = set(NUMPY_NAMES) | func.mod.np_names
        scope = func
        while scope is not None:
            self.np_names |= scope.np_names
            scope = scope.parent
        self.binding_names = set()
        self.entry_names = {}
        self.obj_types = {}   # local name -> the mojolearn class it was constructed from
        self.local_modules = {}   # local name -> module a helper returned (`P = _prep()`)
        self.binding_names |= index.binding_params.get(func.key, set())
        nodes = sorted((n for n in _own_nodes(func.node) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.withitem))),
                       key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
        for _ in range(2):
            for node in nodes:
                value = node.context_expr if isinstance(node, ast.withitem) else node.value
                if value is None:
                    continue
                targets = ([node.optional_vars] if isinstance(node, ast.withitem)
                           else node.targets if isinstance(node, ast.Assign) else [node.target])
                for target in targets:
                    if isinstance(target, ast.Tuple) and isinstance(value, ast.Call):
                        made = self.made_tuple(value)
                        for elt, klass in zip(target.elts, made or ()):
                            if isinstance(elt, ast.Name) and klass is not None:
                                self.obj_types[elt.id] = klass
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if not names:
                    continue
                if self.is_binding(value):
                    self.binding_names.update(names)
                    continue
                entry = self.entry_of(value)
                if entry:
                    for name in names:
                        self.entry_names[name] = entry
                    continue
                made = self.made_class(value)
                if made is not None:
                    for name in names:
                        self.obj_types[name] = made
                    continue
                if isinstance(value, ast.Call) and not isinstance(value.func, ast.Call):
                    target = self.resolve_call(value.func)
                    if target and target[0] == "func" and target[1].key in index.returns_module:
                        for name in names:
                            self.local_modules[name] = index.returns_module[target[1].key]
        if func.parent is not None:
            parent = Scope(index, func.parent, root_cls)
            self.binding_names |= parent.binding_names
            for name, entry in parent.entry_names.items():
                self.entry_names.setdefault(name, entry)
            for name, made in parent.obj_types.items():
                self.obj_types.setdefault(name, made)
            for name, dotted in parent.local_modules.items():
                self.local_modules.setdefault(name, dotted)

    def made_class(self, value):
        """The mojolearn class an expression constructs or a helper returns, if known."""
        if isinstance(value, ast.Name):
            return self.obj_types.get(value.id)
        if not isinstance(value, ast.Call) or isinstance(value.func, ast.Call):
            return None
        if isinstance(value.func, ast.Name) and value.func.id == "cls" and self.func.cls is not None:
            return self.root or self.func.cls
        target = self.resolve_call(value.func)
        if target and target[0] == "class":
            return target[1]
        if target and target[0] == "func":
            return self.index.returns_class.get(target[1].key)
        return None

    def made_tuple(self, value):
        target = self.resolve_call(value.func) if not isinstance(value.func, ast.Call) else None
        if target and target[0] == "func":
            return self.index.returns_tuple.get(target[1].key)
        return None

    def _shadowed(self, name):
        scope = self.func
        while scope is not None:
            if name in scope.local_defs or name in scope.imports:
                return True
            scope = scope.parent
        return False

    def resolve_call(self, func_expr):
        """Resolve a call target with `self`/`cls`/`super()` bound to the root class."""
        index = self.index
        if isinstance(func_expr, ast.Attribute):
            value = func_expr.value
            if isinstance(value, ast.Name) and value.id in ("self", "cls") and self.root is not None:
                return index.lookup(self.root, func_expr.attr)
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "type"
                    and self.root is not None):
                return index.lookup(self.root, func_expr.attr)
            if isinstance(value, ast.Name) and value.id in self.obj_types:
                return index.lookup(self.obj_types[value.id], func_expr.attr)
            if isinstance(value, ast.Name) and value.id in self.local_modules:
                return index.resolve_name(self.local_modules[value.id], func_expr.attr)
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "super"
                    and self.func.cls is not None):
                start = self.root if self.root is not None else self.func.cls
                return index.lookup(start, func_expr.attr, after=self.func.cls)
        return index.resolve_expr(func_expr, self.modname, self.func)

    def is_binding(self, expr):
        if isinstance(expr, ast.IfExp):
            return self.is_binding(expr.body) or self.is_binding(expr.orelse)
        if isinstance(expr, ast.BoolOp):
            return any(self.is_binding(v) for v in expr.values)
        if isinstance(expr, ast.Name):
            if expr.id in self.binding_names:
                return True
            return expr.id in self.func.mod.binding_globals and not self._shadowed(expr.id)
        if isinstance(expr, ast.Attribute):
            if expr.attr == "_module" and self.is_binding(expr.value):
                return True
            if isinstance(expr.value, (ast.Name, ast.Attribute)):
                # an object a `_native_*` bridge module holds (`_native_neural_bridge.RUNTIME`)
                base = self.index.resolve_expr(expr.value, self.modname, self.func)
                if (base and base[0] == "module" and base[1].rsplit(".", 1)[-1].startswith("_native_")
                        and (self.index.resolve_name(base[1], expr.attr) or ("const",))[0] == "const"):
                    return True
                if base and base[0] == "module" and NATIVE_MODULE.match(base[1].rsplit(".", 1)[-1]):
                    return True
            return (isinstance(expr.value, ast.Name) and expr.value.id in ("self", "cls")
                    and expr.attr in self.index.binding_attrs)
        if isinstance(expr, ast.Call):
            name = _final(expr.func)
            if name and (name in GETTER_NAMES or name.endswith("_binding")):
                return True
            target = self.resolve_call(expr.func)
            return bool(target and target[0] == "func" and target[1].key in self.index.getters)
        return False

    def entry_of(self, expr):
        """`b.entry` or getattr(b, "entry") on a binding: the entry name."""
        if isinstance(expr, ast.Attribute) and self.is_binding(expr.value):
            return expr.attr
        if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "getattr"
                and len(expr.args) >= 2 and self.is_binding(expr.args[0])):
            name = expr.args[1]
            return name.value if isinstance(name, ast.Constant) and isinstance(name.value, str) else "<getattr>"
        if isinstance(expr, ast.Call):
            # a helper that returns an entry (`self._prediction_function(name)`)
            target = self.resolve_call(expr.func)
            if target and target[0] == "func" and target[1].key in self.index.entry_getters:
                first = expr.args[0] if expr.args else None
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    return first.value
                return "<" + target[1].node.name + ">"
        return None


# ---------------------------------------------------------------- analysis

class Facts:
    __slots__ = ("statements", "loops", "untagged_loops", "branches", "numpy", "numpy_work", "allocs",
                 "shell", "unresolved", "native_sites", "entries", "callees", "unresolved_names", "loop_sites")

    def __init__(self):
        self.statements = self.loops = self.untagged_loops = self.branches = 0
        self.numpy = self.numpy_work = self.allocs = self.shell = self.unresolved = self.native_sites = 0
        self.entries = set()
        self.callees = []
        self.unresolved_names = []
        self.loop_sites = []


def _check_only(stmts):
    for stmt in stmts:
        if isinstance(stmt, (ast.Raise, ast.Pass)):
            continue
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and _final(stmt.value.func) == "warn"):
            continue
        return False
    return bool(stmts)


def _has_call(stmts):
    return any(isinstance(n, ast.Call) for s in stmts for n in ast.walk(s))


def _literal(expr):
    if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
        return all(isinstance(e, ast.Constant) for e in expr.elts)
    if isinstance(expr, ast.Dict):
        return all(isinstance(e, ast.Constant) for e in (*expr.keys, *expr.values) if e is not None)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "range":
        return all(isinstance(a, ast.Constant) for a in expr.args)
    return False


def _literal_loop(node):
    """A loop over a literal of constants (parameter names, a fixed range): glue, not data work."""
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return _literal(node.iter)
    if isinstance(node, ast.While):
        return False
    return all(_literal(gen.iter) for gen in node.generators)


def _tagged(mod, node):
    first = node.lineno
    last = getattr(node, "end_lineno", first) or first
    lines = [mod.line(first - 1), *(mod.line(i) for i in range(first, min(last, first + 2) + 1))]
    return any(TAG.search(line) for line in lines)


def analyze(index, func, root_cls):
    scope = Scope(index, func, root_cls)
    mod = func.mod
    facts = Facts()
    body = func.node.body
    skip = None
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        skip = body[0]
    for node in _own_nodes(func.node):
        if isinstance(node, ast.stmt) and node is not skip:
            facts.statements += 1
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp)):
            facts.loops += 1
            if not _tagged(mod, node) and not _literal_loop(node):
                facts.untagged_loops += 1
                facts.loop_sites.append(f"{node.lineno}:{mod.line(node.lineno).strip()[:90]}")
        elif isinstance(node, ast.If):
            if not _check_only(node.body) and (_has_call(node.body) or _has_call(node.orelse)):
                facts.branches += 1
        elif isinstance(node, ast.Call):
            _call(index, scope, node, facts)
    return facts


def _call(index, scope, node, facts):
    func_expr = node.func
    name = _final(func_expr)
    # 1. a native entry
    entry = None
    if isinstance(func_expr, ast.Attribute) and scope.is_binding(func_expr.value):
        entry = func_expr.attr
    elif isinstance(func_expr, ast.Name) and func_expr.id in scope.entry_names:
        entry = scope.entry_names[func_expr.id]
    elif isinstance(func_expr, ast.Name) and scope.is_binding(func_expr):
        entry = func_expr.id   # a compiled dispatcher called directly: `_ENTRY(self, "fit", args, _CONTEXT)`
    elif isinstance(func_expr, ast.Call):
        entry = scope.entry_of(func_expr)
    if entry is not None:
        facts.native_sites += 1
        facts.entries.add(entry)
        return
    # 2. binding acquisition: counted nowhere, never walked
    if scope.is_binding(node):
        return
    # 3. NumPy
    root = _root_name(func_expr)
    if root in scope.np_names and root not in (scope.binding_names | set(scope.entry_names)):
        facts.numpy += 1
        attr = name if isinstance(func_expr, ast.Attribute) else root
        if _is_alloc(attr):
            facts.allocs += 1
        if attr not in NUMPY_CONVERSION:
            facts.numpy_work += 1
        return
    target = scope.resolve_call(func_expr)
    if target is None:
        if name and NATIVE_CALL.search(name):
            facts.native_sites += 1
            facts.entries.add(name)
        elif _is_alloc(name) or name in ALLOC_CLASSES:
            facts.allocs += 1
        elif name and not (isinstance(func_expr, ast.Name) and name in BUILTINS) and name not in PLAIN_METHODS:
            facts.unresolved += 1
            facts.unresolved_names.append(f"{node.lineno}:{ast.unparse(func_expr)[:60]}")
        return
    kind, obj = target
    if kind == "module":
        facts.unresolved += 1
        return
    shell = obj.mod.short in SHELL_MODULES and obj.mod.name.count(".") == 1
    if kind == "class":
        if obj.name in ALLOC_CLASSES or shell:
            facts.allocs += obj.name in ALLOC_CLASSES
            facts.shell += 1
            return
        init = index.lookup(obj, "__init__")
        if init and init[0] == "func":
            facts.callees.append((init[1], obj))
        return
    if _is_alloc(name):
        facts.allocs += 1
    if shell:
        facts.shell += 1
        return
    if obj.key in index.getters or obj.key in index.entry_getters:
        return
    root_cls = scope.root
    if (isinstance(func_expr, ast.Attribute) and isinstance(func_expr.value, ast.Name)
            and func_expr.value.id in scope.obj_types):
        root_cls = scope.obj_types[func_expr.value.id]
    elif obj.cls is not None and (root_cls is None or obj.cls not in index.mro(root_cls)):
        root_cls = obj.cls
    facts.callees.append((obj, root_cls))


def _is_alloc(name):
    return bool(name) and (name in ALLOC_NAMES or bool(ALLOC_PATTERN.search(name)))


class Row:
    __slots__ = ("method", "public", "where", "native_calls", "native_entries", "statements", "loops",
                 "untagged_loops", "branches", "numpy", "numpy_work", "allocs", "helpers", "unresolved",
                 "shell", "verdict", "family")


def explain(root, package, public, method):
    """Print the walk of one public method: every function reached and its facts."""
    index = Index(root, package)
    for canonical, origin, aliases in public_names(root, package):
        if public not in aliases and canonical != public:
            continue
        module, _, name = origin.rpartition(".")
        target = index.resolve_name(module, name)
        if target is None:
            print("unresolved", origin)
            return 1
        if target[0] == "func":
            start, klass = target[1], None
        else:
            found = index.lookup(target[1], method)
            if not found or found[0] != "func":
                print("no method", method, "on", origin)
                return 1
            start, klass = found[1], target[1]
        seen, stack = set(), [(start, klass, 0)]
        while stack:
            item, cls, depth = stack.pop()
            key = (item.key, cls.key if cls else None)
            if key in seen:
                continue
            seen.add(key)
            f = analyze(index, item, cls)
            print(f"{'  ' * min(depth, 12)}{item.mod.name}.{item.qual}:{item.node.lineno} stmts={f.statements} "
                  f"native={f.native_sites}{sorted(f.entries) if f.entries else ''} loops={f.loops}/"
                  f"{f.untagged_loops} branches={f.branches} np={f.numpy}/{f.numpy_work} alloc={f.allocs} "
                  f"shell={f.shell}")
            for unresolved in f.unresolved_names:
                print(f"{'  ' * min(depth, 12)}  ? {unresolved}")
            for loop in f.loop_sites:
                print(f"{'  ' * min(depth, 12)}  L {loop}")
            stack.extend((c, k, depth + 1) for c, k in reversed(f.callees))
        return 0
    print("not a public name:", public)
    return 1


def measure(index, func, root_cls, cache):
    """Walk the call path from `func`; aggregate facts over every function reached once."""
    seen, stack = set(), [(func, root_cls)]
    total = Facts()
    helpers = -1
    while stack:
        item, klass = stack.pop()
        key = (item.key, klass.key if klass else None)
        if key in seen:
            continue
        seen.add(key)
        helpers += 1
        if key not in cache:
            cache[key] = analyze(index, item, klass)
        facts = cache[key]
        for slot in ("statements", "loops", "untagged_loops", "branches", "numpy", "numpy_work", "allocs",
                     "shell", "unresolved", "native_sites"):
            setattr(total, slot, getattr(total, slot) + getattr(facts, slot))
        total.entries |= facts.entries
        stack.extend(facts.callees)
    return total, helpers


def verdict(facts, helpers=0):
    if (facts.native_sites == 0 and helpers == 0 and facts.statements <= 3 and not facts.loops
            and not facts.numpy and not facts.allocs and not facts.unresolved and not facts.shell):
        return "native"   # a no-op, e.g. `fit` of a stateless transform that returns self
    if facts.native_sites == 0 or facts.numpy_work or facts.untagged_loops:
        return "python"
    # A call into a Python helper is never native, however thin the helper:
    # the contract is the method body itself making the one native call.
    if (facts.native_sites == 1 and helpers == 0 and not facts.loops and not facts.branches
            and not facts.allocs):
        return "native"
    return "orchestrated"


# ---------------------------------------------------------------- exports and families

def public_names(root, package="mojolearn"):
    """[(public name, origin 'module.name', [all public aliases])], one per symbol origin."""
    inventory = _load_inventory()
    base = root / "python" / package
    rows = {}
    manifest = root / "packaging/native_only/manifest.json"
    skip = set()
    if manifest.is_file():
        for row in json.loads(manifest.read_text())["inventory"]:
            rows[row["algorithm"]] = row.get("symbol_origin")
            if row.get("status") == "not_algorithm":
                skip.add(row["algorithm"])
    paths = [base / "__init__.py", *sorted(base.glob("_expansion_*.py"))]
    for path in paths:
        if path.is_file():
            for name in inventory.explicit_exports(path):
                rows.setdefault(f"{package}.{name}", None)
    resolve, module_of = inventory.origins(root)
    groups = defaultdict(list)
    for public, origin in sorted(rows.items()):
        if public in skip:
            continue
        if origin is None:
            module, _, name = public.rpartition(".")
            origin = resolve(module, name)
        groups[origin].append(public)
    out = []
    for origin, aliases in sorted(groups.items()):
        canonical = min(aliases, key=lambda n: (n.count("."), len(n), n))
        out.append((canonical, origin, sorted(aliases)))
    return out


FAMILY_ALIASES = {
    "linear": "linear_model", "decomp": "decomposition", "prep": "preprocessing", "trees": "ensemble",
    "ann": "neighbors", "cnn": "neural_network", "metrics_impl": "metrics", "randomforest": "ensemble",
    "extratrees": "ensemble", "solver": "linear_model", "gp": "gaussian_process", "gpc": "gaussian_process",
    "hierarchy": "cluster", "iforest": "ensemble", "spectral": "manifold", "umap": "manifold",
    "mlp": "neural_network", "tsa": "sequence", "arima": "sequence", "forest": "ensemble", "gbdt": "ensemble",
    "ivf": "neighbors", "forecasting": "sequence", "parallel_forecasting": "sequence",
    "parallel_gaussian_process": "gaussian_process", "parallel_model_selection": "model_selection",
    "parallel_training": "training", "model_pool_training": "training", "offload_training": "training",
    "mamba": "neural_inference", "transformer": "neural_inference", "samba": "neural_inference",
    "byte_lm": "language_model", "embedding": "neural_network", "ops": "ops", "importers": "ops",
    "cholesky": "linalg", "linalg_fast": "linalg", "neighbors_reference": "neighbors"}


def family_of(canonical, defined_in, names, layout):
    """The scikit-learn style module a public name belongs to, for the summary."""
    short = canonical.rsplit(".", 1)[-1]
    for module, members in layout.items():
        if short in members:
            return FAMILY_ALIASES.get(module, module)
    for alias in names:
        parts = alias.split(".")
        if len(parts) == 3 and not parts[1].startswith("_"):
            return FAMILY_ALIASES.get(parts[1], parts[1])
    parts = defined_in.split(".")
    leaf = parts[1] if len(parts) > 1 else "root"
    leaf = leaf.lstrip("_")
    for prefix in ("expansion_", "surface_", "x_sequence", "native_", "parallel_"):
        if leaf.startswith(prefix):
            leaf = "sequence" if prefix == "x_sequence" else leaf[len(prefix):]
    leaf = re.sub(r"_(impl|host|bridge|public|adapters|trainer_host|checkpoint)$", "", leaf)
    return FAMILY_ALIASES.get(leaf, leaf)


def _layout(index, package):
    mod = index.mod(package + "._sklearn_modules")
    if mod is None or "LAYOUT" not in mod.aliases:
        return {}
    try:
        return ast.literal_eval(mod.aliases["LAYOUT"])
    except (ValueError, TypeError):
        return {}


def inventory_rows(root, package="mojolearn"):
    index = Index(root, package)
    layout = _layout(index, package)
    cache = {}
    rows = []
    # Group the export rows by the definition they resolve to: two public names
    # for one class (an alias, a lazy CPU-inference name) are measured once.
    by_def, order = {}, []
    for canonical, origin, aliases in public_names(root, package):
        module, _, name = origin.rpartition(".")
        target = index.resolve_name(module, name) if module else None
        if target is None:
            for mod in [m for m in index.mods.values() if m is not None]:
                found = mod.lazy.get(name)
                if found and found[0] == "_DEPRECATED_ALIASES":
                    target = index.resolve_name(mod.name, name)
                    break
        if target is not None and target[0] in ("module", "const"):
            continue
        key = ("unresolved", origin) if target is None else (target[0], target[1].key)
        if key not in by_def:
            by_def[key] = [target, origin, []]
            order.append(key)
        by_def[key][2].extend([canonical, *aliases])
    for key in order:
        target, origin, names = by_def[key]
        names = sorted(set(names))
        canonical = min(names, key=lambda n: (n.count("."), len(n), n))
        defined_in = target[1].mod.name if target is not None else origin.rpartition(".")[0]
        family = family_of(canonical, defined_in, names, layout)
        if target is None:
            rows.append(_row("-", canonical, origin, None, None, 0, "unresolved", family))
            continue
        kind, obj = target
        if kind == "func":
            facts, helpers = measure(index, obj, None, cache)
            rows.append(_row(FUNCTION_OP, canonical, _where(root, obj), facts, None, helpers,
                             verdict(facts, helpers), family))
            continue
        for op in OPS:
            found = index.lookup(obj, op)
            if not found or found[0] != "func" or _refusal(found[1].node):
                continue
            facts, helpers = measure(index, found[1], obj, cache)
            rows.append(_row(op, canonical, _where(root, found[1]), facts, None, helpers,
                             verdict(facts, helpers), family))
    rows.sort(key=lambda r: (r.family, r.public, OPS.index(r.method) if r.method in OPS else -1))
    return rows


def _refusal(node):
    """A method whose body only raises (`save` that is not available) is not an operation."""
    body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    return bool(body) and all(isinstance(s, ast.Raise) for s in body)


def _where(root, func):
    return f"{func.mod.path.relative_to(root).as_posix()}:{func.node.lineno}"


def _row(method, public, where, facts, _unused, helpers, verdict_, family):
    row = Row()
    row.method, row.public, row.where, row.verdict, row.family = method, public, where, verdict_, family
    row.helpers = helpers
    if facts is None:
        for slot in ("native_calls", "native_entries", "statements", "loops", "untagged_loops", "branches",
                     "numpy", "numpy_work", "allocs", "unresolved", "shell"):
            setattr(row, slot, 0)
        return row
    row.native_calls, row.native_entries = facts.native_sites, len(facts.entries)
    row.statements, row.loops, row.untagged_loops = facts.statements, facts.loops, facts.untagged_loops
    row.branches, row.numpy, row.numpy_work, row.allocs = facts.branches, facts.numpy, facts.numpy_work, facts.allocs
    row.unresolved, row.shell = facts.unresolved, facts.shell
    return row


# ---------------------------------------------------------------- output

COLUMNS = ("method", "class", "module:line", "native_calls", "py_statements", "loops", "numpy_calls",
           "allocations", "verdict", "family", "native_entries", "route_branches", "untagged_loops",
           "numpy_work", "helpers", "shell_calls", "unresolved_calls")


def to_tsv(rows):
    lines = ["\t".join(COLUMNS)]
    for r in rows:
        lines.append("\t".join(str(v) for v in (
            r.method, r.public, r.where, r.native_calls, r.statements, r.loops, r.numpy, r.allocs, r.verdict,
            r.family, r.native_entries, r.branches, r.untagged_loops, r.numpy_work, r.helpers, r.shell,
            r.unresolved)))
    return "\n".join(lines) + "\n"


def read_tsv(path):
    lines = Path(path).read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line.strip()]


def summarize(records):
    """records: dicts with method/class/verdict/family/native_calls/py_statements."""
    families = defaultdict(lambda: defaultdict(int))
    for rec in records:
        fam = families[rec["family"]]
        fam["methods"] += 1
        fam[rec["verdict"]] += 1
        fam["native_calls"] += int(rec["native_calls"])
        fam["py_statements"] += int(rec["py_statements"])
        fam["names"] = fam.get("names", 0)
    names = defaultdict(set)
    for rec in records:
        names[rec["family"]].add(rec["class"])
    head = f"{'family':<26}{'names':>6}{'methods':>8}{'native':>8}{'orch':>7}{'python':>8}{'unres':>7}" \
           f"{'nat_calls':>10}{'py_stmts':>10}"
    out = [head]
    total = defaultdict(int)
    for name in sorted(families, key=lambda f: (-families[f]["methods"], f)):
        fam = families[name]
        out.append(f"{name:<26}{len(names[name]):>6}{fam['methods']:>8}{fam['native']:>8}{fam['orchestrated']:>7}"
                   f"{fam['python']:>8}{fam['unresolved']:>7}{fam['native_calls']:>10}{fam['py_statements']:>10}")
        for key in ("methods", "native", "orchestrated", "python", "unresolved", "native_calls", "py_statements"):
            total[key] += fam[key]
    all_names = len({rec["class"] for rec in records})
    out.append(f"{'TOTAL':<26}{all_names:>6}{total['methods']:>8}{total['native']:>8}{total['orchestrated']:>7}"
               f"{total['python']:>8}{total['unresolved']:>7}{total['native_calls']:>10}{total['py_statements']:>10}")
    return "\n".join(out) + "\n"


def records_of(rows):
    return [dict(zip(COLUMNS, line.split("\t"))) for line in to_tsv(rows).splitlines()[1:]]


def check(current, baseline):
    """Problems (empty when the check passes) and notes."""
    def counts(records):
        c = defaultdict(int)
        for rec in records:
            c[rec["verdict"]] += 1
        return c
    now, then = counts(current), counts(baseline)
    non_native_now = sum(now[v] for v in VERDICTS if v != "native")
    non_native_then = sum(then[v] for v in VERDICTS if v != "native")
    problems, notes = [], []
    if non_native_now > non_native_then:
        problems.append(f"non-native public methods grew: {non_native_then} -> {non_native_now}")
    if now["python"] > then["python"]:
        problems.append(f"python public methods grew: {then['python']} -> {now['python']}")
    before = {(r["class"], r["method"]): r["verdict"] for r in baseline}
    for rec in current:
        old = before.get((rec["class"], rec["method"]))
        if old is None:
            if rec["verdict"] != "native":
                notes.append(f"new {rec['verdict']}: {rec['class']}.{rec['method']} ({rec['module:line']})")
        elif RANK.get(rec["verdict"], 9) > RANK.get(old, 9):
            problems.append(f"regressed {old} -> {rec['verdict']}: {rec['class']}.{rec['method']} "
                            f"({rec['module:line']})")
        elif RANK.get(rec["verdict"], 9) < RANK.get(old, 9):
            notes.append(f"improved {old} -> {rec['verdict']}: {rec['class']}.{rec['method']}")
    if non_native_now < non_native_then:
        notes.append(f"non-native shrank {non_native_then} -> {non_native_now}: refresh the baseline with --baseline")
    return problems, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--package", default="mojolearn")
    parser.add_argument("--out", type=Path, help="write the TSV here instead of stdout")
    parser.add_argument("--summary", action="store_true", help="print only the per-family summary")
    parser.add_argument("--baseline", action="store_true", help="rewrite the committed baseline")
    parser.add_argument("--check", action="store_true", help="fail if the non-native count grows")
    parser.add_argument("--baseline-file", type=Path, default=None)
    parser.add_argument("--explain", metavar="NAME.METHOD",
                        help="print the walk of one method, e.g. mojolearn.KMeans.fit")
    args = parser.parse_args(argv)
    if args.explain:
        public, _, method = args.explain.rpartition(".")
        if not public.startswith(args.package + "."):
            public, method = args.explain, FUNCTION_OP
        return explain(args.root, args.package, public, method)
    baseline = args.baseline_file or (args.root / "tools/hooks/public_runtime_baseline.tsv")
    rows = inventory_rows(args.root, args.package)
    tsv = to_tsv(rows)
    records = records_of(rows)
    summary = summarize(records)
    if args.baseline:
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text(tsv)
        sys.stdout.write(summary)
        print(f"baseline: {baseline} ({len(rows)} rows)")
        return 0
    if args.check:
        if not baseline.is_file():
            print(f"public-runtime: no baseline at {baseline}", file=sys.stderr)
            return 2
        problems, notes = check(records, read_tsv(baseline))
        sys.stdout.write(summary)
        for note in notes:
            print("note: " + note)
        for problem in problems:
            print("FAIL: " + problem)
        print("public-runtime: " + ("FAIL" if problems else "ok"))
        return 1 if problems else 0
    if args.out:
        args.out.write_text(tsv)
    elif not args.summary:
        sys.stdout.write(tsv)
    (sys.stdout if args.summary else sys.stderr).write(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
