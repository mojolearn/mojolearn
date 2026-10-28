#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The measure of the public runtime migration (docs/plans/PUBLIC_RUNTIME_MIGRATION.md).

The contract: a public operation hands its arguments to ONE native call that
validates, prepares, executes and produces the result. Python-side validation,
conversion and binding orchestration are product execution too.
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
an unresolved `_native_*(...)` call, or a module-level name bound once at import
to an entry (`_NATIVE = _binding()._module.call`, `getattr(binding, "entry")`, a
package `connect(...)` returning `(entry, context)`, or such a name imported from
a sibling module) unless the function rebinds it. Functions that return a binding
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

Static analysis, Python only: nothing is imported or built. A native-shaped
wrapper is not proof that its native entry never calls Python; runtime callback
probes and native-callee audits remain separate qualification requirements.

  python3 tools/public_runtime_inventory.py                 # TSV on stdout, summary on stderr
  python3 tools/public_runtime_inventory.py --summary       # per-family numbers only
  python3 tools/public_runtime_inventory.py --baseline      # rewrite tools/hooks/public_runtime_baseline.tsv
  python3 tools/public_runtime_inventory.py --check         # require zero prohibited public source paths
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
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

# Public operations returned by factories/framework registration are not in the
# package's __all__. Keep them in the same gate as directly exported estimators.
RETURNED_OPERATIONS = (
    ("adapters._torch_compile", "_Compiled", "__call__"),
    ("importers._program", "ImportedProgram", "run"),
)

VERDICTS = ("native", "orchestrated", "python", "unresolved")
RANK = {v: i for i, v in enumerate(VERDICTS)}

# The conversion and argument-check layer: calls into these are shell calls,
# not walked. Everything else in the package is walked.
SHELL_MODULES = frozenset({"_buffer", "_array", "_arrays", "_cast_in", "_bufcheck", "_sk_validate",
                           "_backend", "_optional_numpy", "_lazy_out", "_version", "exceptions"})

GETTER_NAMES = frozenset({"binding", "_bind", "load_host_module", "load_door_host", "load_set",
                          "_load_member", "_exec_binding"})
# Functions that ARE the conversion layer although they live outside SHELL_MODULES: a call into one is
# a shell call, not walked. `_parallel_pool.public_method_result` applies `_mode._public_result`, the
# public-output conversion every plain method's `_public_output` wrapper applies after the native call,
# to a pooled driver's joined result (2026-10-24, par-kernel-ridge dtype flip). The wrapper is not on
# any walked path; the driver's explicit call must not count as runtime Python either.
SHELL_FUNCTIONS = frozenset({("_parallel_pool", "public_method_result")})
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
# Compiled factories that turn a bound compiled entry into a class-slot descriptor, and
# refuse anything else at runtime: bindings/classical_attributes.mojo
# classical_attribute_descriptor_c requires MethodType over a BuiltinFunctionType and
# returns PyInstanceMethod_New of it. The slot then calls that entry.
NATIVE_DESCRIPTOR_FACTORIES = frozenset({"classical_attribute_descriptor"})
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
        # module-level names bound to a compiled entry (`_NATIVE = _binding()._module.call`),
        # filled by Index._module_globals: name -> entry
        self.entry_globals = {}
        # module-level names holding a connection namespace (`_session_connected =
        # _SESSION_CONNECTIONS["identical"]`, filled by a package function that returns
        # `SimpleNamespace(call=binding.entry, ...)`), filled by Index._module_globals:
        # name -> {field: ("binding",) | ("entry", entry)}
        self.namespace_globals = {}
        self.module_scope = None
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
        self.returns_entry_tuple = {}   # a connect function returning (entry, context): per-element entry
        self.returns_module = {}
        self.returns_namespace = {}   # a connect function returning SimpleNamespace(field=binding.entry, ...)
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
            # `adapters` is the Torch/MAX front-end over the same kernels (the same API in another format): no
            # public operations of its own to inventory, verify or sabotage (Andrew, 2026-10-24; AGENTS.md).
            if "tests" in rel.parts or rel.parts[0] in ("verify_reference", "reference_cards", "adapters"):
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
        modules = [m for m in self.mods.values() if m is not None]
        for _ in range(6):
            before = (len(self.getters), len(self.entry_getters), len(self.binding_attrs), len(self.returns_class),
                      len(self.returns_tuple), len(self.returns_module), len(self.returns_entry_tuple),
                      sum(map(len, self.binding_params.values())),
                      sum(len(m.entry_globals) + len(m.binding_globals) for m in modules),
                      len(self.returns_namespace), sum(len(m.namespace_globals) for m in modules))
            for mod in modules:
                self._module_globals(mod)
            for func in everything:
                scope = Scope(self, func, func.cls)
                fields = self._namespace_return(scope, func)
                if fields:
                    self.returns_namespace[func.key] = fields
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
                        if isinstance(node.value, ast.Tuple) and func.key not in self.returns_entry_tuple:
                            entries = [scope.entry_of(elt) or (elt.id in scope.entry_names and scope.entry_names[elt.id])
                                       if isinstance(elt, ast.Name) else scope.entry_of(elt)
                                       for elt in node.value.elts]
                            if any(entries):
                                self.returns_entry_tuple[func.key] = [e or None for e in entries]
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
                    len(self.returns_entry_tuple), sum(map(len, self.binding_params.values())),
                    sum(len(m.entry_globals) + len(m.binding_globals) for m in modules),
                    len(self.returns_namespace), sum(len(m.namespace_globals) for m in modules)) == before:
                break

    def _namespace_return(self, scope, func):
        """`result = SimpleNamespace(call=module.entry, binding=module, ...); return result`.

        The fields a connect function returns, when every return is that one local, the local
        is bound exactly once to a keyword-only SimpleNamespace, and nothing in the function
        (or a package callee it is handed to) rebinds a field. Only fields whose value is
        statically a binding or a compiled entry are kept; any other field stays unknown, so
        reading it is never counted as native.
        """
        nodes = _own_nodes(func.node)
        returns = [n for n in nodes if isinstance(n, ast.Return)]
        if not returns or not all(isinstance(r.value, ast.Name) for r in returns):
            return None
        name = returns[0].value.id
        if any(r.value.id != name for r in returns):
            return None
        made, stores = [], []
        for node in nodes:
            if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                if any(isinstance(t, ast.Name) and t.id == name for t in ast.walk(node.target)):
                    return None
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                if any(isinstance(t, ast.Name) and t.id == name for t in ast.walk(node.optional_vars)):
                    return None
            elif isinstance(node, ast.NamedExpr) and node.target.id == name:
                return None
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if any(isinstance(t, ast.Name) and t.id == name for t in ast.walk(target)
                           if not isinstance(target, ast.Attribute)):
                        made.append(node)
                    if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                            and target.value.id == name):
                        stores.append((target.attr, node))
            elif isinstance(node, ast.Call) and _final(node.func) in ("setattr", "delattr"):
                if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == name:
                    return None
        if len(made) != 1 or not isinstance(made[0], ast.Assign) or len(made[0].targets) != 1:
            return None
        call = made[0].value
        if (not isinstance(call, ast.Call) or _final(call.func) != "SimpleNamespace" or call.args
                or any(k.arg is None for k in call.keywords)):
            return None

        def kind(expr):
            if scope.is_binding(expr):
                return ("binding",)
            entry = scope.entry_of(expr)
            if not entry and isinstance(expr, ast.Name):
                entry = scope.entry_names.get(expr.id)
            return ("entry", entry) if entry else None

        fields = {k.arg: kind(k.value) for k in call.keywords}
        for attr, node in stores:
            if sum(a == attr for a, _ in stores) > 1:
                fields[attr] = None
            else:
                fields[attr] = kind(node.value) if isinstance(node, ast.Assign) else None
        # a package helper handed the namespace must not rebind its fields either
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            for position, arg in enumerate(node.args):
                if not (isinstance(arg, ast.Name) and arg.id == name):
                    continue
                target = scope.resolve_call(node.func) if not isinstance(node.func, ast.Call) else None
                if not target or target[0] != "func":
                    return None
                params = _params(target[1].node)[0]
                if position >= len(params):
                    return None
                param = params[position]
                for inner in _own_nodes(target[1].node):
                    if isinstance(inner, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
                        for t in (inner.targets if isinstance(inner, (ast.Assign, ast.Delete)) else [inner.target]):
                            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                    and t.value.id == param):
                                return None
                    if (isinstance(inner, ast.Call) and _final(inner.func) in ("setattr", "delattr")
                            and inner.args and isinstance(inner.args[0], ast.Name) and inner.args[0].id == param):
                        return None
        fields = {k: v for k, v in fields.items() if v}
        return fields or None

    def _namespace_globals(self, mod):
        """Module-level names that can only hold one connection namespace.

        `_C = {}`, then `_C[mode] = _connect(mode)` (a returns_namespace function) or a
        refusal string, then `_c = _C["identical"]` (also when rebound under an `if`):
        `_c` holds the namespace. Every module-level binding of the name (top level and inside
        if/for/try/with blocks) must agree, and no function may store into the container.
        """
        assigned = defaultdict(list)   # name -> values bound by `name = value`
        stored = defaultdict(list)     # container -> values stored by `name[k] = value`

        def walk(body):
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            assigned[target.id].append(node.value)
                        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                            stored[target.value.id].append(node.value)
                        else:
                            for t in ast.walk(target):
                                if isinstance(t, ast.Name):
                                    assigned[t.id].append(None)
                elif isinstance(node, (ast.For, ast.AsyncFor)):
                    for t in ast.walk(node.target):
                        if isinstance(t, ast.Name):
                            assigned[t.id].append(None)
                for field in ("body", "orelse", "finalbody"):
                    walk(getattr(node, field, []) or [])
                for handler in getattr(node, "handlers", []) or []:
                    if handler.name:
                        assigned[handler.name].append(None)
                    walk(handler.body)

        walk(mod.tree.body)
        inner_stores = set()
        for node in ast.walk(mod.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for inner in ast.walk(node):
                    if isinstance(inner, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
                        for t in (inner.targets if isinstance(inner, (ast.Assign, ast.Delete)) else [inner.target]):
                            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                                inner_stores.add(t.value.id)
                    elif isinstance(inner, ast.Global):
                        inner_stores.update(inner.names)
        scope = Scope(self, mod.module_scope, None)

        def inert(value):   # a refusal text kept in place of a connection
            return (isinstance(value, ast.JoinedStr) or isinstance(value, ast.Constant) and isinstance(value.value, str)
                    or isinstance(value, ast.Attribute) and value.attr == "__name__"
                    or isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add) and inert(value.left)
                    or isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "str")

        def fields_of(value, containers):
            if isinstance(value, ast.Call) and not isinstance(value.func, ast.Call):
                target = scope.resolve_call(value.func)
                if target and target[0] == "func":
                    return self.returns_namespace.get(target[1].key)
            if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
                return containers.get(value.value.id)
            if isinstance(value, ast.Name):
                return mod.namespace_globals.get(value.id)
            return None

        containers = {}
        for name, values in stored.items():
            if name in inner_stores:
                continue
            if not all(isinstance(v, ast.Dict) and not v.keys for v in assigned.get(name, [])):
                continue
            made = [fields_of(v, {}) for v in values if not inert(v)]
            if made and made[0] and all(m == made[0] for m in made):
                containers[name] = made[0]
        for name, values in assigned.items():
            if name in stored or name in inner_stores or name in mod.defs or name in mod.classes:
                continue
            made = [fields_of(v, containers) if v is not None else None for v in values]
            if made and made[0] and all(m == made[0] for m in made):
                mod.namespace_globals[name] = made[0]

    def _module_globals(self, mod):
        """Module-level names bound once at import to a binding or a compiled entry.

        `_NATIVE = _binding()._module.small_mlp_call`, `_E = getattr(_backend.binding(...), "x")`,
        `_E = _connect()` where the package function `_connect` returns an entry (or a binding), and
        `from ._impl import _NATIVE` of such a name in another module. Calling the name is then a
        native entry call, exactly like calling `b.entry(...)` inside the method.
        """
        if mod.module_scope is None:
            node = ast.FunctionDef(name="<module>", args=ast.arguments(
                posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[s for s in mod.tree.body if isinstance(s, (ast.Assign, ast.AnnAssign))],
                decorator_list=[], returns=None, type_comment=None, lineno=1, col_offset=0)
            mod.module_scope = Func(mod, node)
        self._namespace_globals(mod)
        scope = Scope(self, mod.module_scope, None)
        mod.binding_globals |= scope.binding_names
        for name, entry in scope.entry_names.items():
            mod.entry_globals.setdefault(name, entry)
        for local, imported in mod.imports.items():
            if imported[0] != "from" or local in mod.entry_globals or local in mod.binding_globals:
                continue
            source = self.mod(imported[1]) if self.module_file(imported[1]) else None
            if source is None or source is mod:
                continue
            if imported[2] in source.entry_globals:
                mod.entry_globals[local] = source.entry_globals[imported[2]]
            elif imported[2] in source.binding_globals:
                mod.binding_globals.add(local)


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
        # names this function binds itself (parameters, assignments): they shadow module-level
        # bindings and entries unless the local value is one too
        self.rebound = set()
        if func.node.name != "<module>":
            self.rebound.update(_params(func.node)[1])
            for node in nodes:
                if isinstance(node, ast.withitem):
                    targets = [node.optional_vars] if node.optional_vars is not None else []
                else:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    elts = target.elts if isinstance(target, ast.Tuple) else [target]
                    self.rebound.update(t.id for t in elts if isinstance(t, ast.Name))
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
                        # `_ENTRY, _CONTEXT = connect(...)`: a package function returning (entry, context)
                        called = self.resolve_call(value.func) if not isinstance(value.func, ast.Call) else None
                        if called and called[0] == "func":
                            for elt, entry in zip(target.elts, index.returns_entry_tuple.get(called[1].key, ())):
                                if isinstance(elt, ast.Name) and entry:
                                    self.entry_names[elt.id] = entry
                    elif (isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple)
                          and len(target.elts) == len(value.elts)):
                        # `_E, _C = b._module.call, context`
                        for elt, part in zip(target.elts, value.elts):
                            if not isinstance(elt, ast.Name):
                                continue
                            if self.is_binding(part):
                                self.binding_names.add(elt.id)
                            elif self.entry_of(part):
                                self.entry_names[elt.id] = self.entry_of(part)
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
        # a module-level name bound to a compiled entry, unless this function rebinds it
        if func.mod.entry_globals:
            for name, entry in func.mod.entry_globals.items():
                if name not in self.rebound and not self._shadowed(name) and name not in self.binding_names:
                    self.entry_names.setdefault(name, entry)
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
            return (expr.id in self.func.mod.binding_globals and not self._shadowed(expr.id)
                    and expr.id not in self.rebound)
        if isinstance(expr, ast.Attribute):
            if expr.attr == "_module" and self.is_binding(expr.value):
                return True
            if self._namespace_field(expr) == ("binding",):
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

    def _namespace_field(self, expr):
        """The field of a module-level connection namespace (`_session_connected.call`)."""
        value = expr.value
        if not isinstance(value, ast.Name):
            return None
        fields = self.func.mod.namespace_globals.get(value.id)
        if not fields or value.id in self.rebound or self._shadowed(value.id) or value.id in self.binding_names:
            return None
        return fields.get(expr.attr)

    def entry_of(self, expr):
        """`b.entry` or getattr(b, "entry") on a binding: the entry name."""
        if isinstance(expr, ast.Attribute):
            field = self._namespace_field(expr)
            if field and field[0] == "entry":
                return field[1]
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
    elif isinstance(func_expr, ast.Attribute) and isinstance(func_expr.value, ast.Name):
        # `_selection._NATIVE(None, "family_msel_parallel_score", ...)`: a sibling module's own
        # module-level compiled dispatcher or entry, reached through `from . import model_selection
        # as _selection` (parallel_model_selection.cross_val_score, 2026-10-24: native, not Python).
        base = index.resolve_expr(func_expr.value, scope.modname, scope.func)
        if base and base[0] == "module" and index.module_file(base[1]):
            sibling = index.mod(base[1])
            if sibling is not None and sibling is not scope.func.mod:
                if func_expr.attr in sibling.entry_globals:
                    entry = sibling.entry_globals[func_expr.attr]
                elif func_expr.attr in sibling.binding_globals:
                    entry = func_expr.attr
    if entry is not None:
        facts.native_sites += 1
        facts.entries.add(entry)
        return
    # 2. binding acquisition: counted nowhere, never walked
    if scope.is_binding(node):
        if getattr(index, "strict_endpoints", False):
            facts.shell += 1  # runtime binding acquisition is product orchestration
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
        if name and NATIVE_CALL.search(name) and not getattr(index, "strict_endpoints", False):
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
    connections = getattr(index, "endpoint_connections", None)
    if connections is not None:
        owner = None
        if isinstance(func_expr, ast.Attribute):
            receiver = func_expr.value
            if isinstance(receiver, ast.Name):
                owner = scope.root if receiver.id in ("self", "cls") else scope.obj_types.get(receiver.id)
            if owner is None:
                receiver_target = index.resolve_expr(receiver, scope.modname, scope.func)
                if receiver_target and receiver_target[0] == "class":
                    owner = receiver_target[1]
        replaced = connections.get(owner, name) if owner and isinstance(func_expr, ast.Attribute) else []
        if kind == "class":
            replaced = connections.get(obj, "__init__")
        if replaced:
            values = [r["value"] for r in replaced]
            value = values[0] if values and values[0] and all(v == values[0] for v in values) else None
            if value and value[0] == "native":
                facts.native_sites += 1
                facts.entries.add(value[1])
                return
            if value and value[0] == "func":
                kind, obj = value
            else:
                facts.unresolved += 1
                facts.unresolved_names.append(f"{node.lineno}:unresolved installed slot {name}")
                return
        if kind == "func":
            decorated = connections.decorated(obj)
            if decorated and decorated[0] == "func":
                obj = decorated[1]
            elif decorated and decorated[0] == "native":
                facts.native_sites += 1
                facts.entries.add(decorated[1])
                return
            else:
                facts.unresolved += 1
                facts.unresolved_names.append(f"{node.lineno}:unresolved decorator on {name}")
                return
    shell = obj.mod.short in SHELL_MODULES and obj.mod.name.count(".") == 1
    if kind == "func" and (obj.mod.short, obj.qual) in SHELL_FUNCTIONS and obj.mod.name.count(".") == 1:
        shell = True
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
    if (facts.native_sites == 1 and helpers == 0 and facts.statements == 1
            and not facts.loops and not facts.branches and not facts.allocs
            and not facts.shell and not facts.numpy and not facts.unresolved):
        return "native"
    return "orchestrated"


# ---------------------------------------------------------------- exports and families

def public_names(root, package="mojolearn", include_non_algorithm=False):
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
        if public in skip and not include_non_algorithm:
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
    "parallel_ensemble": "ensemble", "parallel_graph": "manifold", "parallel_neighbors": "neighbors",
    "parallel_neighbors_reference": "neighbors", "parallel_preprocessing": "preprocessing",
    "parallel_classical": "multi_gpu",
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
    for module, name, op in RETURNED_OPERATIONS:
        path = Path(root) / "python" / package / (module.replace(".", "/") + ".py")
        if not path.is_file():
            continue
        public = f"{package}.{module}.{name}"
        target = index.resolve_name(f"{package}.{module}", name)
        found = index.lookup(target[1], op) if target and target[0] == "class" else None
        if not found or found[0] != "func":
            rows.append(_row(op, public, str(path.relative_to(root)), None, None, 0,
                             "unresolved", "returned_operations"))
            continue
        facts, helpers = measure(index, found[1], target[1], cache)
        rows.append(_row(op, public, _where(root, found[1]), facts, None, helpers,
                         verdict(facts, helpers), "returned_operations"))
    rows.sort(key=lambda r: (r.family, r.public, OPS.index(r.method) if r.method in OPS else -1))
    return rows


def _public_member(name):
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def _endpoint_parameters(node):
    """Keep option names, kinds and defaults, without evaluating product code."""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    out = []
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        if index == 0 and arg.arg in ("self", "cls"):
            continue
        out.append(dict(name=arg.arg, kind="positional_only" if index < len(args.posonlyargs)
                        else "positional_or_keyword", required=default is None,
                        default=ast.unparse(default) if default is not None else None))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out.append(dict(name=arg.arg, kind="keyword_only", required=default is None,
                        default=ast.unparse(default) if default is not None else None))
    for arg, kind in ((args.vararg, "var_positional"), (args.kwarg, "var_keyword")):
        if arg:
            out.append(dict(name=arg.arg, kind=kind, required=False, default=None))
    return out


class EndpointConnections:
    """Resolve import-time class-slot replacements without importing the product.

    This small symbolic walk follows concrete class/function identities through
    installers and decorators. It does not execute calls or infer that a value
    returned by a native factory must itself be native. Unknown installed slots
    remain qualification debt; their superseded Python bodies are not reported
    as live execution. Existing runtime evidence must still prove transitivity.
    """

    def __init__(self, index):
        self.index, self.replacements, self.seen = index, {}, set()
        self.native_flags = set()   # classes an installer marks `cls._mojolearn_native_public = True`
        self.scopes = {}
        for cls in list(index.classes.values()):
            for node in cls.node.body:
                if isinstance(node, ast.Assign):
                    value = self.value(node.value, cls.mod, None, {})
                    if value and value[0] in ("native", "func"):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.record(cls, target.id, value, cls.mod, node, "class-definition slot connection")
        for mod in list(index.mods.values()):
            if mod is None:
                continue
            self.walk(mod.tree.body, mod, None, {})
        for cls in list(index.classes.values()):
            for decorator in reversed(cls.node.decorator_list):
                generated = _dataclass_methods(cls, decorator)
                if generated is not None:
                    end = getattr(cls.node, "end_lineno", cls.node.lineno)
                    own = cls.mod.path.relative_to(self.index.root).as_posix() + ":"
                    for name in generated:
                        # a module-level replacement after the class statement supersedes it
                        later = [r for r in self.replacements.get((cls.key, name), [])
                                 if r["source"].startswith(own) and int(r["source"].rsplit(":", 1)[1]) > end]
                        if not later:
                            self.record(cls, name, ("python", "dataclasses.dataclass"), cls.mod, decorator,
                                        "dataclass-generated Python method")
                    continue
                value = self.value(decorator, cls.mod, None, {})
                if value and value[0] == "func":
                    self.call(value[1], [("class", cls)], [], {})
                else:
                    self.record(cls, "*", None, cls.mod, decorator,
                                "unresolved class decorator can replace public operations")

    def scope(self, func):
        if func.key not in self.scopes:
            self.scopes[func.key] = Scope(self.index, func, func.cls)
        return self.scopes[func.key]

    def value(self, expr, mod, func, env, depth=0):
        if expr is None or depth > 12:
            return None
        if isinstance(expr, ast.Constant):
            return ("literal", expr.value)
        if isinstance(expr, ast.Name):
            if expr.id in env:
                return env[expr.id]
            target = self.index.resolve_expr(expr, mod.name, func)
            if target and target[0] != "const":
                return target
            scope = self.scope(func or mod.module_scope) if func or mod.module_scope else None
            if scope and expr.id in scope.entry_names:
                return ("native", scope.entry_names[expr.id])
            alias = mod.aliases.get(expr.id)
            if alias is not None and not isinstance(alias, ast.Call):
                return self.value(alias, mod, func, env, depth + 1)
        if isinstance(expr, (ast.Tuple, ast.List)):
            return ("sequence", [self.value(e, mod, func, env, depth + 1) for e in expr.elts])
        if isinstance(expr, ast.Dict) and None not in expr.keys:
            # a constant table keyed by literals (`_LIFECYCLE_METHODS[(module, name)]`)
            keys = [_canonical(self.value(k, mod, func, env, depth + 1)) for k in expr.keys]
            if all(k is not None for k in keys):
                return ("table", tuple(zip(keys, expr.values)), mod, func)
            return None
        if isinstance(expr, ast.Subscript):
            if (isinstance(expr.value, ast.Attribute) and expr.value.attr == "__dict__"):
                # `cls.__dict__["fit"]`: the class's own namespace entry, never a base's
                owner = self.value(expr.value.value, mod, func, env, depth + 1)
                key = self.value(expr.slice, mod, func, env, depth + 1)
                if owner and owner[0] == "class" and key and key[0] == "literal":
                    for node in owner[1].node.body:
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == key[1]:
                            return ("func", self.index.func(owner[1].mod, node, cls=owner[1]))
                return None
            table = self.value(expr.value, mod, func, env, depth + 1)
            key = _canonical(self.value(expr.slice, mod, func, env, depth + 1))
            if table and table[0] == "table" and key is not None:
                for candidate, value in table[1]:
                    if candidate == key:
                        return self.value(value, table[2], table[3], {}, depth + 1)
            return None
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            # `(base,) + tuple(classes)`: a concatenation of known sequences
            left = self.value(expr.left, mod, func, env, depth + 1)
            right = self.value(expr.right, mod, func, env, depth + 1)
            if left and right and left[0] == right[0] == "sequence":
                return ("sequence", left[1] + right[1])
            return None
        if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in ("tuple", "list")
                and len(expr.args) == 1 and not expr.keywords):
            inner = self.value(expr.args[0], mod, func, env, depth + 1)
            if inner and inner[0] == "sequence":
                return inner
        if isinstance(expr, ast.Attribute):
            if (isinstance(expr.value, ast.Name) and expr.value.id == "object"
                    and expr.attr in ("__getattribute__", "__setattr__", "__delattr__", "__new__", "__init__")):
                return ("native", "object." + expr.attr)
            base = self.value(expr.value, mod, func, env, depth + 1)
            if base and base[0] == "class" and expr.attr in ("__module__", "__name__", "__qualname__"):
                # a package class statement: its module and names are fixed by the source
                if not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == expr.attr
                                                             for t in n.targets) for n in base[1].node.body):
                    return ("literal", base[1].mod.name if expr.attr == "__module__" else base[1].name)
            if base and base[0] == "class":
                return self.index.lookup(base[1], expr.attr)
            if base and base[0] == "func" and expr.attr in ("__wrapped__", "__func__"):
                return base
            target = self.index.resolve_expr(expr, mod.name, func)
            if target:
                return target
        scope = self.scope(func or mod.module_scope) if func or mod.module_scope else None
        if scope:
            entry = scope.entry_of(expr)
            if entry:
                return ("native", entry)
        literal = self.literal_test(expr, mod, func, env, depth)
        if literal is not None:
            return literal
        if isinstance(expr, ast.Call):
            args = [self.value(a, mod, func, env, depth + 1) for a in expr.args]
            if (_final(expr.func) in ("getattr", "getattr_static") and len(args) >= 2
                    and args[0] and args[0][0] == "class" and args[1] and args[1][0] == "literal"):
                found = self.index.lookup(args[0][1], args[1][1])
                if (found is None and len(args) == 3 and args[2] and args[2][0] == "literal"
                        and isinstance(args[1][1], str) and not args[1][1].startswith("__")
                        and self.complete_mro(args[0][1])):
                    # `getattr(cls, "decision_function", None)` on a class whose whole MRO is
                    # package source that never defines the name: the default.
                    return args[2]
                return found
            if _final(expr.func) == "unwrap" and args:
                return args[0]
            factory = self.value(expr.func, mod, func, env, depth + 1)
            if (factory and factory[0] == "native" and factory[1] in NATIVE_DESCRIPTOR_FACTORIES
                    and len(expr.args) == 1 and isinstance(expr.args[0], ast.Call)
                    and _final(expr.args[0].func) == "MethodType" and expr.args[0].args):
                bound = self.value(expr.args[0].args[0], mod, func, env, depth + 1)
                if bound and bound[0] == "native":
                    return bound
            if (isinstance(expr.func, ast.Name) and expr.func.id in ("classmethod", "staticmethod")
                    and len(args) == 1 and args[0] and args[0][0] == "func"):
                # `cls.load = classmethod(inspect.unwrap(loader.__func__))`: the same source body
                return args[0]
            called = self.index.resolve_expr(expr.func, mod.name, func)
            if called and called[0] == "func":
                # A decorator factory returns a source function, not a native
                # operation just because its name contains 'native'.
                callee = called[1]
                local = dict(zip(_params(callee.node)[0], args))
                returns = [self.value(n.value, callee.mod, callee, local, depth + 1)
                           for n in _own_nodes(callee.node) if isinstance(n, ast.Return)]
                if returns and returns[0] and all(r == returns[0] for r in returns):
                    return returns[0]
        return None

    def complete_mro(self, cls):
        """Every base on the MRO resolves to package source (or is `object`)."""
        for klass in self.index.mro(cls):
            for base in klass.node.bases:
                if isinstance(base, ast.Name) and base.id == "object":
                    continue
                resolved = self.index.resolve_expr(base, klass.mod.name, None)
                if not resolved or resolved[0] != "class":
                    return False
            if klass.node.keywords:
                return False
        return True

    def literal_test(self, expr, mod, func, env, depth):
        """A condition decided by literal values only: `kind == "iforest"`, `kind in _KINDS`,
        `inspect.isfunction(method)` on a looked-up function or a literal default, `not x`.
        None when anything in it is not a known literal, so both branches stay walked."""
        if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
            left = self.value(expr.left, mod, func, env, depth + 1)
            right = self.value(expr.comparators[0], mod, func, env, depth + 1)
            if not left or left[0] != "literal" or not right:
                return None
            op = expr.ops[0]
            if right[0] == "literal" and isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                same = left[1] == right[1] if isinstance(op, (ast.Eq, ast.NotEq)) else left[1] is right[1]
                return ("literal", same if isinstance(op, (ast.Eq, ast.Is)) else not same)
            if (right[0] == "sequence" and isinstance(op, (ast.In, ast.NotIn))
                    and all(v and v[0] == "literal" for v in right[1])):
                inside = any(left[1] == v[1] for v in right[1])
                return ("literal", inside if isinstance(op, ast.In) else not inside)
            return None
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            inner = self.value(expr.operand, mod, func, env, depth + 1)
            if inner and inner[0] == "literal":
                return ("literal", not inner[1])
            return None
        if isinstance(expr, ast.BoolOp):
            values = [self.value(v, mod, func, env, depth + 1) for v in expr.values]
            if all(v and v[0] == "literal" for v in values):
                truth = [bool(v[1]) for v in values]
                return ("literal", all(truth) if isinstance(expr.op, ast.And) else any(truth))
            return None
        if (isinstance(expr, ast.Call) and _final(expr.func) in ("isfunction", "callable")
                and len(expr.args) == 1 and not expr.keywords):
            arg = self.value(expr.args[0], mod, func, env, depth + 1)
            if arg and arg[0] == "literal":
                return ("literal", callable(arg[1]))
            if (arg and arg[0] == "func" and not arg[1].node.decorator_list
                    and isinstance(arg[1].node, ast.FunctionDef)):
                return ("literal", True)
        return None

    def record(self, cls, name, value, mod, node, reason="import-time class-slot replacement"):
        if name != "*" and not _public_member(name):
            return
        row = dict(value=value, source=f"{mod.path.relative_to(self.index.root)}:{node.lineno}",
                   expression=ast.unparse(node), reason=reason)
        key = (cls.key, name)
        current = self.replacements.setdefault(key, [])
        if not any(r["source"] == row["source"] and r["expression"] == row["expression"] for r in current):
            current.append(row)

    def assign(self, target, value, mod, func, env, node):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and value and value[0] == "sequence":
            for dst, src in zip(target.elts, value[1]):
                self.assign(dst, src, mod, func, env, node)
        elif isinstance(target, ast.Attribute):
            cls = self.value(target.value, mod, func, env)
            if cls and cls[0] == "class":
                if target.attr == "_mojolearn_native_public":
                    self.native_flags.add(cls[1].key)
                self.record(cls[1], target.attr, value, mod, node)

    def call(self, func, args, keywords, parent_env):
        env = dict(zip(_params(func.node)[0], args))
        env.update(keywords)
        # Closures keep the concrete class connected by their enclosing factory.
        env = dict(parent_env, **env)
        identity = tuple(sorted((k, v[0], v[1].key if v[0] in ("class", "func") else str(v[1])[:120])
                                for k, v in env.items() if v and v[0] in ("class", "func", "literal")))
        key = (func.key, identity)
        if key in self.seen:
            return
        self.seen.add(key)
        self.walk(func.node.body, func.mod, func, env)

    def walk(self, body, mod, func, env):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = self.value(node.value, mod, func, env)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    self.assign(target, value, mod, func, env, node)
            if isinstance(node, ast.For):
                seq = self.value(node.iter, mod, func, env)
                if seq and seq[0] == "sequence":
                    for value in seq[1]:
                        local = dict(env)
                        self.assign(node.target, value, mod, func, local, node)
                        self.walk(node.body, mod, func, local)
                else:
                    self.walk(node.body, mod, func, dict(env))
                self.walk(node.orelse, mod, func, dict(env))
                continue
            if isinstance(node, ast.If):
                test = self.value(node.test, mod, func, env)
                if test and test[0] == "literal":
                    # a branch decided by literals (`elif kind == "iforest":` for one registered
                    # class): the installers of the other branch never run for it
                    self.walk(node.body if test[1] else node.orelse, mod, func, dict(env))
                    continue
            if isinstance(node, (ast.If, ast.Try, ast.With)):
                self.walk(node.body, mod, func, dict(env))
                self.walk(getattr(node, "orelse", []), mod, func, dict(env))
                self.walk(getattr(node, "finalbody", []), mod, func, dict(env))
                for handler in getattr(node, "handlers", []):
                    self.walk(handler.body, mod, func, dict(env))
                continue
            # Expressions in assignments can invoke installers too.
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                args = [self.value(a, mod, func, env) for a in call.args]
                if _final(call.func) == "setattr" and len(args) >= 3 and args[0] and args[0][0] == "class":
                    name = args[1][1] if args[1] and args[1][0] == "literal" and isinstance(args[1][1], str) else "*"
                    self.record(args[0][1], name, args[2], mod, call)
                called = self.index.resolve_expr(call.func, mod.name, func)
                if called and called[0] == "func":
                    # Package import roots plus helpers carrying a concrete
                    # class. Do not walk ordinary product calls as installers.
                    has_class = any(a and (a[0] == "class" or a[0] == "sequence" and
                        any(v and v[0] == "class" for v in a[1])) for a in args)
                    if func is None or has_class:
                        keywords = [(k.arg, self.value(k.value, mod, func, env)) for k in call.keywords if k.arg]
                        self.call(called[1], args, keywords, env if called[1].parent else {})

    def get(self, cls, name):
        for owner in self.index.mro(cls):
            rows = self.replacements.get((owner.key, name), [])
            if name != "*":
                rows = rows + self.replacements.get((owner.key, "*"), [])
            if rows:
                return rows
            # An own source definition shadows a base's installed attribute.
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
                   or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
                   for n in owner.node.body):
                break
        return []

    def decorated(self, func):
        current = ("func", func)
        for decorator in reversed(func.node.decorator_list):
            name = ast.unparse(decorator.func if isinstance(decorator, ast.Call) else decorator)
            if name in ("property", "classmethod", "staticmethod", "functools.wraps", "wraps") or name.endswith((".getter", ".setter", ".deleter")):
                continue
            factory = self.value(decorator, func.mod, func.parent, {})
            if not factory or factory[0] != "func":
                return None
            outer = factory[1]
            env = dict(zip(_params(outer.node)[0], [current]))
            returns = [self.value(n.value, outer.mod, outer, env) for n in _own_nodes(outer.node)
                       if isinstance(n, ast.Return)]
            if not returns or not returns[0] or any(v != returns[0] for v in returns):
                return None
            current = returns[0]
        return current


def _dataclass_methods(cls, decorator):
    """The Python methods `@dataclass(...)` with literal options generates on `cls`, or None
    when the decorator is not a literal dataclass call (the class's own definitions are kept;
    a frozen class always gets the generated __setattr__/__delattr__)."""
    call = decorator if isinstance(decorator, ast.Call) else None
    target = call.func if call else decorator
    if ast.unparse(target) not in ("dataclass", "dataclasses.dataclass"):
        return None
    imported = cls.mod.imports.get("dataclass") if isinstance(target, ast.Name) else ("mod", "dataclasses")
    if not imported or (imported[0] == "from" and imported[1] != "dataclasses"):
        return None
    options = dict(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
    for keyword in (call.keywords if call else []):
        if keyword.arg not in options or not isinstance(keyword.value, ast.Constant):
            return None
        options[keyword.arg] = bool(keyword.value.value)
    own = {n.name for n in cls.node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    own |= {t.id for n in cls.node.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    names = []
    for enabled, generated in ((options["init"], ("__init__",)), (options["repr"], ("__repr__",)),
                               (options["eq"], ("__eq__",)),
                               (options["order"], ("__lt__", "__le__", "__gt__", "__ge__"))):
        if enabled:
            names += [n for n in generated if n not in own]
    if options["frozen"]:
        names += ["__setattr__", "__delattr__"]
    if (options["unsafe_hash"] or options["eq"] and options["frozen"]) and "__hash__" not in own:
        names.append("__hash__")
    return names


def _canonical(value):
    """A hashable form of a literal or a sequence of literals, for table keys."""
    if value and value[0] == "literal":
        return ("literal", value[1])
    if value and value[0] == "sequence":
        items = [_canonical(v) for v in value[1]]
        return ("sequence", tuple(items)) if all(i is not None for i in items) else None
    return None


def _endpoint_verdict(func, facts, helpers):
    if (facts.unresolved and facts.statements == 1 and not facts.loops
            and not facts.numpy and not facts.allocs and not facts.shell):
        return "unresolved"
    result = verdict(facts, helpers)
    if result == "native" and facts.native_sites == 0:
        body = [n for n in func.node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        # Merely being short is not a native boundary. In particular, property
        # hooks, scalar coercions and constructor assignments execute Python.
        noop = all(isinstance(n, ast.Pass) or isinstance(n, ast.Return) and
                   (n.value is None or isinstance(n.value, ast.Constant) and n.value.value is None
                    or isinstance(n.value, ast.Name) and n.value.id == "self") for n in body)
        if not noop:
            return "python"
    return result


def _tag_metadata_only(index, func, owner):
    """The tag name does not exempt calls into ordinary product operations."""
    computational = set(OPS) | {"kneighbors", "radius_neighbors", "search", "run", "backward"}
    seen, pending = set(), [(func, owner)]
    while pending:
        current, cls = pending.pop()
        key = (current.key, cls.key if cls else None)
        if key in seen:
            continue
        seen.add(key)
        for node in _own_nodes(current.node):
            if isinstance(node, ast.Call) and _final(node.func) in computational:
                return False
        pending.extend(analyze(index, current, cls).callees)
    return True


class ClassHookWrappers:
    """Python wrappers `_mode`'s class-creation hooks put around source methods.

    `PublicOutputsMixin.__init_subclass__` (install_public_outputs) and
    `SklearnTagsMixin.__init_subclass__` (install_fit_tracking) replace a class's plain
    function methods with Python wrappers (`_public_output`, `_track_fit`, `_needs_fit`,
    `_loader`) unless the class is `_mojolearn_native_public` when it is created. The
    source body then is not the installed callable: the operation runs Python
    validation and output conversion around it. Conservative in the reporting
    direction: a decorated method (decorators can carry the skip markers), a class an
    installer later marks native (or any of its bases), and a slot an installer
    replaces are never reported.
    """

    def __init__(self, index, connections, package):
        self.index, self.connections = index, connections
        self.ready = False
        mode = index.mod(package + "._mode")
        if mode is None:
            return
        names = {}
        for name in ("_OUTPUTS", "_FITS", "_NEEDS_FIT", "_LOADS"):
            try:
                names[name] = frozenset(ast.literal_eval(mode.aliases[name]))
            except (KeyError, ValueError, TypeError, SyntaxError):
                return
        outputs = index.resolve_name(mode.name, "PublicOutputsMixin")
        tags = index.resolve_name(mode.name, "SklearnTagsMixin")
        if not outputs or not tags or outputs[0] != "class" or tags[0] != "class":
            return
        self.names, self.outputs, self.tags, self.ready = names, outputs[1], tags[1], True

    def _flagged(self, cls):
        for klass in self.index.mro(cls):
            if klass.key in self.connections.native_flags:
                return True
            for node in klass.node.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_mojolearn_native_public"
                                                        for t in node.targets):
                    return not (isinstance(node.value, ast.Constant) and not node.value.value)
        return False

    def _reaches_hooks(self, klass):
        """Creating `klass` runs the _mode hooks: every `__init_subclass__` before them on
        the MRO hands on with `super().__init_subclass__(...)`."""
        for base in self.index.mro(klass)[1:]:
            if base in (self.outputs, self.tags):
                return True
            hook = next((n for n in base.node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and n.name == "__init_subclass__"), None)
            if hook is None:
                continue
            if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr == "__init_subclass__" and isinstance(n.func.value, ast.Call)
                       and _final(n.func.value.func) == "super" for n in ast.walk(hook)):
                return False
        return False

    def wrapper(self, owner, name, func):
        """The class whose creation wrapped `owner.name` (resolving to source `func`), or None."""
        if not self.ready or owner is None or func is None or func.node.decorator_list:
            return None
        if self.connections.get(owner, name) or self._flagged(owner):
            return None
        names = self.names
        for klass in self.index.mro(owner):
            chain = self.index.mro(klass)
            if klass in (self.outputs, self.tags) or self.outputs not in chain:
                continue
            if self._flagged(klass) or not self._reaches_hooks(klass):
                continue
            found = self.index.lookup(klass, name)
            if not found or found[0] != "func" or found[1] is not func:
                continue
            own = func.cls is klass
            if name in names["_OUTPUTS"]:
                return klass
            if self.tags in chain and (name in names["_NEEDS_FIT"] or name in names["_FITS"] and own):
                return klass
        return None


def endpoint_inventory(root, package="mojolearn", installed_slots=None):
    """One static ledger for public operations and their omitted lifecycle surface.

    Evidence is deliberately unclaimed: a syntactic reference or an algorithm's
    lane cannot establish that a constructor, option or failure path executed.
    Runtime installers and external base classes require a separate runtime walk.
    """
    witnessed, matched_slots = {}, set()
    if installed_slots is not None:
        import installed_attribute_slots as slot_evidence
        if package != "mojolearn":
            raise ValueError("installed-slot evidence currently models mojolearn only")
        witnessed = slot_evidence.validate(installed_slots, root)
    index, cache, rows, by_definition = Index(root, package), {}, [], {}
    layout = _layout(index, package)
    for canonical, origin, aliases in public_names(root, package, include_non_algorithm=True):
        module, _, name = origin.rpartition(".")
        target = index.resolve_name(module, name) if module else None
        if target and target[0] in ("module", "const"):
            continue
        key = (target[0], target[1].key) if target else ("unresolved", origin)
        group = by_definition.setdefault(key, dict(target=target, origin=origin, names=set()))
        group["names"].update([canonical, *aliases])
    # Factory-produced objects are product APIs even when their class is private
    # or the export manifest calls them 'not_algorithm'. Retain the explicit
    # framework-returned doors whose registration factories are not exports.
    for module, name, _ in RETURNED_OPERATIONS:
        dotted = f"{package}.{module}"
        if index.module_file(dotted) is None:
            continue
        target = index.resolve_name(dotted, name)
        key = (target[0], target[1].key) if target else ("unresolved", dotted + "." + name)
        by_definition.setdefault(key, dict(target=target, origin=dotted + "." + name,
                                           names={dotted + "." + name}, returned=True))
    for group in list(by_definition.values()):
        target = group["target"]
        if target and target[0] == "func":
            cls = index.returns_class.get(target[1].key)
            if cls:
                key = ("class", cls.key)
                by_definition.setdefault(key, dict(target=("class", cls), origin=".".join(cls.key),
                    names={min(group["names"]) + "::<result>"}, returned=True))
    connections = EndpointConnections(index)
    index.endpoint_connections = connections
    index.strict_endpoints = True
    wrappers = ClassHookWrappers(index, connections, package)

    def add(group, method, kind, func=None, owner=None):
        names = sorted(group["names"])
        public = min(names, key=lambda n: (n.count("."), len(n), n))
        defined_in = func.mod.name if func else group["origin"].rpartition(".")[0]
        item = dict(endpoint=public + "." + method, public=public, aliases=names,
                    method=method, kind=kind, definition=None, source_ast_sha256=None,
                    parameters=[], static_verdict="unresolved", native_entries=[],
                    family=family_of(public, defined_in, names, layout),
                    returned_operation=group.get("returned", False),
                    runtime_status="unresolved", qualification_debt=[],
                    evidence={axis: {"status": "unreviewed", "records": []} for axis in
                              ("runtime", "correctness", "identity", "fault", "options", "failures")})
        declared = func
        replacement = connections.get(owner, method.split(":", 1)[0]) if owner else []
        if replacement:
            item["import_connections"] = [{k: v for k, v in r.items() if k != "value"} for r in replacement]
            item["declared_definition"] = _where(root, func) if func else None
            values = [r["value"] for r in replacement]
            if values and values[0] and all(v == values[0] for v in values):
                if values[0][0] == "func":
                    func = values[0][1]
                elif values[0][0] == "native":
                    func = None
                    item.update(static_verdict="native", native_entries=[values[0][1]],
                                runtime_status="native_boundary")
                elif values[0][0] == "python":
                    # a Python implementation generated at class creation (dataclass)
                    func = None
                    item.update(static_verdict="python", runtime_status="prohibited_source",
                                generated_by=values[0][1], definition=item["import_connections"][0]["source"])
                else:
                    func = None
            else:
                func = None
            if func is None and item["static_verdict"] != "native":
                item["qualification_debt"].append("import-time replacement requires installed-slot qualification")
        if func:
            decorated = connections.decorated(func)
            if decorated and decorated[0] == "func":
                func = decorated[1]
            elif decorated and decorated[0] == "native":
                item.update(static_verdict="native", native_entries=[decorated[1]], runtime_status="native_boundary")
                func = None
            else:
                item["qualification_debt"].append("unresolved method decorator; source body is not the installed callable")
                item["declared_definition"] = _where(root, func)
                func = None
        if func:
            # Descriptors share a source name; avoid the ordinary method cache
            # collapsing their getter, setter and deleter into one definition.
            facts, helpers = measure(index, func, owner, cache)
            item.update(definition=_where(root, func),
                        source_ast_sha256=hashlib.sha256(ast.dump(func.node).encode()).hexdigest(),
                        parameters=_endpoint_parameters(func.node),
                        static_verdict=_endpoint_verdict(func, facts, helpers), native_entries=sorted(map(str, facts.entries)),
                        always_refuses=_refusal(func.node),
                        unresolved_calls=facts.unresolved, shell_calls=facts.shell)
            item["runtime_status"] = ("native_boundary" if item["static_verdict"] == "native" else
                                      "unresolved" if item["static_verdict"] == "unresolved" else "prohibited_source")
            wrapped = wrappers.wrapper(owner, method, func) if kind in ("method", "method_alias") and not replacement else None
            if wrapped is not None:
                # the installed callable is a _mode Python wrapper around this body
                item.update(static_verdict="python", runtime_status="prohibited_source",
                            class_creation_wrapper=".".join(wrapped.key))
        if method == "__sklearn_tags__" and func is not None and _tag_metadata_only(index, func, owner):
            item["runtime_status"] = "allowed_tag_metadata"
            item["exception"] = "Owner-authorized scikit-learn tag metadata only; native callbacks remain prohibited."
        elif item.get("always_refuses"):
            item["runtime_status"] = "refusal_only"
        elif method == "__init_subclass__" and item["static_verdict"] != "native":
            item["runtime_status"] = "unresolved"
            item["qualification_debt"].append("class-definition hook requires import-time versus product-call boundary qualification")
        if declared and replacement and func is not declared:
            item["declared_source_ast_sha256"] = hashlib.sha256(ast.dump(declared.node).encode()).hexdigest()
        if owner and witnessed and method == "__getattribute__":
            class_key = ".".join(owner.key)
            slot_evidence.apply(item, class_key, witnessed)
            if "installed_slot_witness" in item:
                matched_slots.add(class_key)
        rows.append(item)

    for group in by_definition.values():
        target = group["target"]
        if target is None:
            add(group, "?", "unresolved_export")
            continue
        kind, obj = target
        if kind == "func":
            add(group, FUNCTION_OP, "function", obj)
            continue
        seen = set()
        for cls in index.mro(obj):
            members = {}
            for node in cls.node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public_member(node.name):
                    members.setdefault(node.name, []).append(node)
                elif isinstance(node, ast.Assign):
                    for name in node.targets:
                        if isinstance(name, ast.Name) and _public_member(name.id):
                            members.setdefault(name.id, [])
            for name, nodes in members.items():
                if name in seen:
                    continue
                seen.add(name)
                found = index.lookup(obj, name)
                if not nodes:
                    if found and found[0] == "func":
                        add(group, name, "method_alias", found[1], obj)
                    elif connections.get(obj, name):
                        add(group, name, "installed_method", owner=obj)
                    else:
                        assigned = next((n.value for n in cls.node.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)), None)
                        if isinstance(assigned, ast.Call) or isinstance(assigned, (ast.Name, ast.Attribute)) and not found:
                            # A custom descriptor or imported callable cannot
                            # disappear just because lookup cannot resolve it.
                            add(group, name, "unresolved_class_attribute", owner=obj)
                    continue
                for node in nodes:
                    decorators = [ast.unparse(d) for d in node.decorator_list]
                    access = next((d.rsplit(".", 1)[-1] for d in decorators
                                   if d == "property" or d.endswith((".setter", ".getter", ".deleter"))), None)
                    if access:
                        suffix = {"property": "get", "getter": "get", "setter": "set", "deleter": "delete"}[access]
                        func = Func(cls.mod, node, cls=cls)
                        func.qual += ":" + suffix
                        func.key = (func.mod.name, func.qual)
                        add(group, name + ":" + suffix, "property", func, obj)
                    elif found and found[0] == "func":
                        add(group, name, "constructor" if name in ("__init__", "__new__") else "method", found[1], obj)
                        break
        for cls in index.mro(obj):
            for cls_key, name in connections.replacements:
                if cls_key == cls.key and name not in seen:
                    seen.add(name)
                    add(group, name, "unresolved_installer" if name == "*" else "installed_method", owner=obj)
            for base in cls.node.bases:
                if isinstance(base, ast.Name) and base.id in ("object", "Exception", "ValueError", "RuntimeError"):
                    continue
                resolved = index.resolve_expr(base, cls.mod.name, None)
                if not resolved or resolved[0] != "class":
                    add(group, "<base:" + ast.unparse(base) + ">", "unresolved_base")
            if cls.node.keywords:
                add(group, "<class-construction>", "unresolved_metaclass")
    if set(witnessed) != matched_slots:
        raise ValueError("installed-slot classes do not resolve to current public getters: "
                         + ", ".join(sorted(set(witnessed) - matched_slots)))
    rows.sort(key=lambda row: row["endpoint"])
    return dict(schema=2, scope="all public and returned endpoints; source analysis plus optional retained installed-slot identity, no product imports or execution",
                limitations=["Dynamic native installers and external-base methods need runtime enumeration.",
                             "Static native-shaped calls do not prove transitive callees avoid Python.",
                             "Parameters describe declared options, not tested values or combinations.",
                             "Evidence starts unreviewed; existing results must be linked with their exact scope."],
                endpoints=rows,
                summary=dict(endpoints=len(rows), constructors=sum(r["kind"] == "constructor" for r in rows),
                             properties=sum(r["kind"] == "property" for r in rows),
                             refusal_only=sum(r.get("always_refuses", False) for r in rows),
                             declared_parameters=sum(len(r["parameters"]) for r in rows),
                             by_kind={k: sum(r["kind"] == k for r in rows) for k in sorted({r["kind"] for r in rows})},
                             by_status={k: sum(r["runtime_status"] == k for r in rows) for k in sorted({r["runtime_status"] for r in rows})},
                             installed_attribute_slots=len(matched_slots)))


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
    """Require zero non-native operation rows; the baseline is history, never permission."""
    def counts(records):
        c = defaultdict(int)
        for rec in records:
            c[rec["verdict"]] += 1
        return c
    now, then = counts(current), counts(baseline)
    non_native_now = sum(now[v] for v in VERDICTS if v != "native")
    non_native_then = sum(then[v] for v in VERDICTS if v != "native")
    problems, notes = [], []
    if non_native_now:
        problems.append(f"non-native public operations remain: {non_native_now}; zero required "
                        "regardless of baseline")
        for rec in current:
            if rec["verdict"] != "native":
                problems.append(f"{rec['verdict']}: {rec['class']}.{rec['method']} "
                                f"({rec['module:line']})")
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


def check_endpoints(report):
    """Zero prohibited OR unresolved live endpoints, independent of OPS/history."""
    admitted = {"native_boundary", "allowed_tag_metadata", "refusal_only"}
    problems = []
    for row in report["endpoints"]:
        if row["runtime_status"] not in admitted:
            detail = "; ".join(row.get("qualification_debt", [])) or row["static_verdict"]
            problems.append(f"{row['runtime_status']}: {row['endpoint']} "
                            f"({row.get('definition') or row.get('declared_definition') or 'unresolved source'}): {detail}")
    return problems


def summarize_endpoints(report):
    families = defaultdict(lambda: defaultdict(int))
    for row in report["endpoints"]:
        families[row["family"]][row["runtime_status"]] += 1
    fields = ("native_boundary", "prohibited_source", "unresolved", "allowed_tag_metadata", "refusal_only")
    lines = [f"{'family':<28} {'native':>7} {'prohibited':>10} {'unresolved':>10} {'tags':>6} {'refusal':>8}"]
    for family, counts in sorted(families.items()):
        lines.append(f"{family:<28} " + " ".join(f"{counts[k]:>10}" for k in fields))
    lines.append("endpoint kinds: " + json.dumps(report["summary"]["by_kind"], sort_keys=True))
    lines.append("endpoint status: " + json.dumps(report["summary"]["by_status"], sort_keys=True))
    return "\n".join(lines) + "\n"


def endpoint_frontier(report):
    """Group gate debt by live implementation or unresolved connection site."""
    groups, classes, kinds, families = {}, defaultdict(int), defaultdict(int), defaultdict(lambda: defaultdict(int))
    for row in report["endpoints"]:
        status = row["runtime_status"]
        if status not in ("prohibited_source", "unresolved"):
            continue
        classes[row["public"]] += 1
        kinds[row["kind"]] += 1
        families[row["family"]][status] += 1
        reasons = row.get("qualification_debt") or [row["static_verdict"]]
        locations = ([r["source"] for r in row.get("import_connections", [])] if status == "unresolved" else [])
        locations = sorted(set(locations or [row.get("definition") or row.get("declared_definition") or row["public"]]))
        key = (status, tuple(locations), tuple(reasons))
        group = groups.setdefault(key, dict(status=status, locations=locations, reasons=reasons, endpoints=[], kinds=set(), families=set()))
        group["endpoints"].append(row["endpoint"])
        group["kinds"].add(row["kind"])
        group["families"].add(row["family"])
    result = []
    for group in groups.values():
        result.append(dict(group, kinds=sorted(group["kinds"]), families=sorted(group["families"]), count=len(group["endpoints"])))
    result.sort(key=lambda g: (-g["count"], g["locations"]))
    return dict(implementation_groups=result, unique_implementation_groups=len(result),
                by_class=dict(sorted(classes.items())), by_kind=dict(sorted(kinds.items())),
                by_family={k: dict(v) for k, v in sorted(families.items())})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--package", default="mojolearn")
    parser.add_argument("--out", type=Path, help="write the TSV here instead of stdout")
    parser.add_argument("--summary", action="store_true", help="print only the per-family summary")
    parser.add_argument("--endpoints-json", type=Path,
                        help="write all static public lifecycle/option endpoints and unclaimed evidence axes")
    parser.add_argument("--installed-slots", type=Path,
                        help="admit an exact-source, receipted installed-attribute witness (classification only)")
    parser.add_argument("--baseline", action="store_true", help="rewrite the committed baseline")
    parser.add_argument("--check", action="store_true", help="require zero prohibited/unresolved public or returned endpoints")
    parser.add_argument("--baseline-file", type=Path, default=None)
    parser.add_argument("--explain", metavar="NAME.METHOD",
                        help="print the walk of one method, e.g. mojolearn.KMeans.fit")
    args = parser.parse_args(argv)
    if args.installed_slots and not (args.endpoints_json or args.check):
        parser.error("--installed-slots requires --endpoints-json or --check")
    if args.endpoints_json or args.check:
        if args.baseline or args.explain:
            parser.error("endpoint qualification is separate from the historical operation baseline")
        try:
            report = endpoint_inventory(args.root, args.package, args.installed_slots)
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
            print("public-runtime evidence refused: " + str(error), file=sys.stderr)
            return 2
        problems = check_endpoints(report)
        report["gate"] = dict(status="failed" if problems else "source-classification-passed",
                              prohibited_or_unresolved=len(problems), problems=problems,
                              runtime_proof_required=True)
        report["frontier"] = endpoint_frontier(report)
        if args.endpoints_json:
            args.endpoints_json.write_text(json.dumps(report, indent=2) + "\n")
        if args.check:
            sys.stdout.write(summarize_endpoints(report))
            for problem in problems[:40]:
                print("FAIL: " + problem)
            if len(problems) > 40:
                print(f"{len(problems) - 40} additional failures; retain every endpoint with --endpoints-json PATH")
            print("public-runtime: " + (f"FAIL ({len(problems)} prohibited/unresolved endpoints; zero required)"
                  if problems else "source classification passed; transitive runtime proof still required"))
            return 1 if problems else 0
        print(json.dumps(report["summary"], sort_keys=True))
        return 0
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
    if args.out:
        args.out.write_text(tsv)
    elif not args.summary:
        sys.stdout.write(tsv)
    (sys.stdout if args.summary else sys.stderr).write(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
