#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source-derive the field-only constructor schemas of the declared-setter
families (the prep `_PrepBase` and decomp `_Base` classes whose `set_params`
is the shallow declared assignment, python/mojolearn/_native_declared_params.py).

Static analysis only: nothing is imported. A constructor qualifies when its
resolved `__init__` body is a docstring plus assignments whose targets are
`self.<field>` (or a tuple of them) and whose values are constructor parameter
names or literal constants (or a matching tuple of them). Anything else
(calls, super(), arithmetic, conditionals, *args/**kwargs) keeps the class out.
None of these families is a NumericModeMixin, so there is no wrapper rule.

  python3 tools/declared_constructor_schemas.py            # print the schema literal
  python3 tools/declared_constructor_schemas.py --refused  # list classes left in Python and why
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from public_runtime_inventory import Index  # noqa: E402

FAMILIES = (("mojolearn._expansion_prep", "_PrepBase"), ("mojolearn._expansion_decomp", "_Base"))
# Plain public classes outside those families whose constructors only store
# their parameters (found by the same rule; none is a NumericModeMixin).
# The resample results are not here: their constructors are one call to the
# resample owner's native result initializer (family_rs_result_initialize,
# providers/python/resample_public_pyapi.mojo), which admits releases itself.
EXTRAS = (("mojolearn.model_selection", ("ShuffleSplit", "GroupShuffleSplit", "StratifiedShuffleSplit",
                                         "LeavePOut", "LeavePGroupsOut")),)


def _all(mod):
    for node in mod.tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    return ()


def _defines(cls, name):
    return next((n for n in cls.node.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _bind_super(call, base_init, value_of):
    """Map a `super().__init__(...)` call's arguments onto the base
    constructor's parameters: {base parameter: (kind, value)}, or None."""
    args = base_init.args
    if args.vararg or args.kwarg or args.posonlyargs or base_init.decorator_list:
        return None
    positional = [a.arg for a in args.args[1:]]
    kwonly = [a.arg for a in args.kwonlyargs]
    bound = {}
    if len(call.args) > len(positional) or any(isinstance(a, ast.Starred) for a in call.args):
        return None
    for name, node in zip(positional, call.args):
        bound[name] = value_of(node)
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in bound or keyword.arg not in positional + kwonly:
            return None
        bound[keyword.arg] = value_of(keyword.value)
    defaults = dict(zip(positional[len(positional) - len(args.defaults):], args.defaults))
    defaults.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None})
    for name in positional + kwonly:
        if name not in bound:
            if name not in defaults:
                return None
            bound[name] = value_of(defaults[name], constant_only=True)
    if any(value is None for value in bound.values()):
        return None
    return bound


def _events(init, params, base=None):
    """The ordered write events of a field-only body, or (None, reason).

    `base` resolves the next constructor for a `super().__init__(...)`
    statement: () -> (FunctionDef, its own base resolver) or None. Its
    events are inlined with the call's arguments substituted."""
    events = []

    def value_of(node, constant_only=False):
        if not constant_only and isinstance(node, ast.Name) and node.id in params:
            return params[node.id] if isinstance(params, dict) else ("parameter", node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool, type(None))):
            return ("constant", node.value)
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float))
                and not isinstance(node.operand.value, bool)):
            return ("constant", -node.operand.value)
        return None

    def field(node):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            return node.attr
        return None

    body = list(init.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    for statement in body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "__init__"
                and isinstance(call.func.value, ast.Call) and isinstance(call.func.value.func, ast.Name)
                and call.func.value.func.id == "super" and not call.func.value.args and base is not None):
            resolved = base()
            if resolved is None:
                return None, f"line {statement.lineno}: super().__init__ has no source constructor"
            base_init, base_base = resolved
            bound = _bind_super(call, base_init, value_of)
            if bound is None:
                return None, f"line {statement.lineno}: super().__init__ arguments are not names or constants"
            inner, reason = _events(base_init, bound, base_base)
            if inner is None:
                return None, reason
            events.extend(inner)
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return None, f"line {statement.lineno}: not a single field assignment"
        target, value = statement.targets[0], statement.value
        if isinstance(target, ast.Tuple):
            if not isinstance(value, ast.Tuple) or len(value.elts) != len(target.elts):
                return None, f"line {statement.lineno}: unpacking is not a literal tuple of equal length"
            pairs = list(zip(target.elts, value.elts))
        else:
            pairs = [(target, value)]
        # A tuple assignment evaluates every value first, then assigns left
        # to right; values here are names/constants, so order is the targets'.
        for dst, src in pairs:
            name, source = field(dst), value_of(src)
            if name is None or source is None:
                return None, f"line {statement.lineno}: {ast.unparse(statement)}"
            events.append((name, source[0], source[1], False))
    return tuple(events), None


def _candidates(index):
    for modname, basename in FAMILIES:
        mod = index.mod(modname)
        kind, base = index.resolve_name(modname, basename)
        setter = _defines(base, "set_params")
        for name in _all(mod):
            target = index.resolve_name(modname, name)
            if not target or target[0] != "class":
                continue
            cls = target[1]
            owner = next((c for c in index.mro(cls) if _defines(c, "set_params")), None)
            if owner is not base or _defines(owner, "set_params") is not setter:
                continue
            yield modname, name, cls
    for modname, names in EXTRAS:
        for name in names:
            target = index.resolve_name(modname, name)
            if target and target[0] == "class":
                if any(c.name == "NumericModeMixin" for c in index.mro(target[1])):
                    raise ValueError(f"{name}: a NumericModeMixin needs the wrapper rule")
                yield modname, name, target[1]


def derive(root=ROOT):
    index = Index(root)
    schemas, refused = {}, {}
    for modname, name, cls in _candidates(index):
        mro = index.mro(cls)
        defining = next((c for c in mro if _defines(c, "__init__")), None)
        if defining is None:
            refused[name] = "no source constructor"
            continue
        init = _defines(defining, "__init__")
        args = init.args
        if args.vararg or args.kwarg or args.posonlyargs or init.decorator_list:
            refused[name] = "variadic, positional-only or decorated constructor"
            continue
        params = [a.arg for a in args.args[1:]] + [a.arg for a in args.kwonlyargs]

        def resolver(after):
            def base():
                later = mro[mro.index(after) + 1:]
                found = next((c for c in later if _defines(c, "__init__")), None)
                return None if found is None else (_defines(found, "__init__"), resolver(found))
            return base
        events, reason = _events(init, set(params), resolver(defining))
        if events is None:
            refused[name] = reason
            continue
        schemas[name] = dict(module=modname, defining=f"{defining.mod.name}.{defining.name}",
                             line=init.lineno, parameters=tuple(params), events=events,
                             ast_dump=ast.dump(init, include_attributes=False))
    return schemas, refused


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    schemas, refused = derive()
    if "--refused" in argv:
        for name, reason in sorted(refused.items()):
            print(f"{name}: {reason}")
        return 0
    print("_EVENTS = {")
    for name in sorted(schemas):
        print(f"    {name!r}: {schemas[name]['events']!r},")
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
