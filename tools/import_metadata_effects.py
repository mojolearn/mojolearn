# SPDX-License-Identifier: Apache-2.0
"""Conservative source proof for import-time class-registry connectors.

Unknown consumers, class constructors/methods and protocol-bearing objects are
not metadata operations. Comments and connector names confer no exemption.
"""

import ast
from dataclasses import dataclass, replace
from pathlib import PurePosixPath


@dataclass(frozen=True)
class Value:
    kind: str = "unknown"
    item: object = None
    hot: bool = False
    slots: tuple = ()


UNKNOWN = Value()
SCALAR = Value("scalar")
AMBIGUOUS = ast.Name(id="__unproved_metadata_binding__", ctx=ast.Load())


def scope_statements(body):
    """Walk control flow, never the body of a nested function or class."""
    for node in body:
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for _field, value in ast.iter_fields(node):
                if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                    yield from scope_statements(value)


def merge(values):
    values = list(values)
    if not values:
        return UNKNOWN
    first = values[0]
    hot = any(v.hot for v in values)
    if any(v.kind == "derived" for v in values):
        return Value("derived", hot=True)
    if all(v.kind == first.kind and v.item == first.item for v in values):
        return replace(first, hot=hot)
    if all(v.kind == "dict" for v in values):
        return Value("dict", merge(v.item for v in values), hot)
    if all(v.kind == "tuple" and len(v.item) == len(first.item) for v in values):
        return Value("tuple", tuple(merge(v.item[i] for v in values) for i in range(len(first.item))), hot)
    return Value("unknown", hot=hot)


class Unsafe(Exception):
    pass


class Source:
    def __init__(self, parse, native_factory, registry_guard=None, call_site=None):
        self.parse = parse
        self.native_factory = native_factory
        self.registry_guard = registry_guard
        self.call_site = call_site
        self.busy = set()

    def registry_unchanged(self, rel, name):
        """A literal alone does not prove the shape of a mutated registry."""
        tree = self.parse(rel)
        aliases = {name}
        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in aliases
                ):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id not in aliases:
                            aliases.add(target.id)
                            changed = True
        mutators = []
        for statement in tree.body:
            if self.call_site and rel == self.call_site[0] and statement.lineno >= self.call_site[1]:
                continue
            writes = False
            for node in ast.walk(statement):
                if isinstance(node, ast.Global) and aliases.intersection(node.names):
                    writes = True
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                    if any(
                        isinstance(part, ast.Name) and part.id in aliases for part in ast.walk(node.value)
                    ):
                        writes = True
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id in aliases:
                        writes = True
                if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in ("tuple", "list", "dict", "set", "frozenset", "len")
                        and self.lookup(rel, node.func.id) is None
                    ):
                        continue
                    if isinstance(node, ast.Call) and any(
                        isinstance(part, ast.Name) and part.id in aliases
                        for arg in node.args + [kw.value for kw in node.keywords]
                        for part in ast.walk(arg)
                    ):
                        writes = True
            if writes:
                mutators.append(statement)
        return not mutators or bool(self.registry_guard and self.registry_guard(rel, name, mutators))

    def module_path(self, rel, module, level=0):
        if level:
            parent = PurePosixPath(rel).parent
            for _ in range(level - 1):
                parent = parent.parent
            path = str(parent.joinpath(*(module or "").split(".")))
        elif module == "mojolearn" or module.startswith("mojolearn."):
            path = "python/" + module.replace(".", "/")
        else:
            return None
        for candidate in (path + ".py", path + "/__init__.py"):
            if self.parse(candidate) is not None:
                return candidate
        return None

    def lookup(self, rel, name, body=None, seen=None):
        seen = set() if seen is None else seen
        if (rel, name) in seen:
            return None
        seen.add((rel, name))
        tree = self.parse(rel)
        if tree is None:
            return None

        # A local binding shadows the module binding. Multiple writes or a
        # conditional binding have no proved single identity.
        def bindings(statements):
            result = []
            for node in scope_statements(statements):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name == name
                ):
                    result.append(node)
                elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                    (alias.asname or alias.name) == name for alias in node.names
                ):
                    result.append(node)
                elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(
                        isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == name
                        for target in targets
                        for n in ast.walk(target)
                    ):
                        result.append(node)
                elif isinstance(node, (ast.For, ast.AsyncFor)) and any(
                    isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == name
                    for n in ast.walk(node.target)
                ):
                    result.append(node)
            return result

        nodes = bindings(body or [])
        direct = body or []
        if not nodes:
            nodes, direct = bindings(tree.body), tree.body
        if len(nodes) > 1 or any(node not in direct for node in nodes):
            return rel, AMBIGUOUS
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return rel, node
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) != name:
                        continue
                    if node.module == "types" and alias.name in (
                        "MethodType",
                        "BuiltinFunctionType",
                        "GetSetDescriptorType",
                    ):
                        return "types", alias.name
                    module = self.module_path(
                        rel, ".".join(v for v in (node.module, alias.name) if v), node.level
                    )
                    if module:
                        return module, None
                    module = self.module_path(rel, node.module or "", node.level)
                    if module:
                        return self.lookup(module, alias.name, seen=seen)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        module = self.module_path(rel, alias.name)
                        if module:
                            return module, None
            if isinstance(node, ast.Assign):
                for dest in node.targets:
                    if isinstance(dest, ast.Name) and dest.id == name:
                        return rel, node.value
                    if isinstance(dest, (ast.Tuple, ast.List)):
                        for index, element in enumerate(dest.elts):
                            if isinstance(element, ast.Name) and element.id == name:
                                return rel, ast.Subscript(
                                    value=node.value, slice=ast.Constant(index), ctx=ast.Load()
                                )
        return (rel, AMBIGUOUS) if nodes else None

    def target(self, rel, node, body=None):
        if isinstance(node, ast.Name):
            return self.lookup(rel, node.id, body)
        if isinstance(node, ast.Attribute):
            parent = self.target(rel, node.value, body)
            if parent and parent[1] is None:
                return self.lookup(parent[0], node.attr)
        return None

    def plain_class(self, rel, node, seen=None):
        seen = set() if seen is None else seen
        key = rel, node.name
        if key in seen:
            return False
        seen.add(key)
        if node.decorator_list or node.keywords:
            return False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "object" and self.lookup(rel, "object") is None:
                continue
            target = self.target(rel, base)
            if (
                not target
                or not isinstance(target[1], ast.ClassDef)
                or not self.plain_class(*target, seen.copy())
            ):
                return False
        return True

    def shape(self, rel, node, env=None, body=None, depth=0):
        if depth > 15:
            return UNKNOWN
        if node is AMBIGUOUS:
            return UNKNOWN
        env = {} if env is None else env
        if isinstance(node, ast.Constant):
            return SCALAR
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            target = self.lookup(rel, node.id, body)
            if target:
                if target[0] == "types":
                    return Value("type_marker", target[1])
                if isinstance(target[1], ast.ClassDef):
                    return Value("class" if self.plain_class(*target) else "hook_class")
                if isinstance(target[1], ast.FunctionDef):
                    return Value("function", target)
                if target[1] is None:
                    return Value("module", target[0])
                if isinstance(target[1], ast.expr):
                    if isinstance(target[1], ast.Dict) and not self.registry_unchanged(target[0], node.id):
                        return UNKNOWN
                    key = target[0], node.id
                    if key in self.busy:
                        return UNKNOWN
                    self.busy.add(key)
                    try:
                        return self.shape(
                            target[0], target[1], env if target[0] == rel else {}, None, depth + 1
                        )
                    finally:
                        self.busy.remove(key)
            return UNKNOWN
        if isinstance(node, ast.Dict):
            # Registry keys resolve to literal strings/numbers. An arbitrary
            # key can run __hash__/__eq__ when a copied registry is built.
            if any(
                key is None or self.shape(rel, key, env, body, depth + 1).kind != "scalar"
                for key in node.keys
            ):
                return UNKNOWN
            values = [self.shape(rel, value, env, body, depth + 1) for value in node.values]
            slots = tuple(
                (key.value, value) for key, value in zip(node.keys, values) if isinstance(key, ast.Constant)
            )
            return Value("dict", merge(values), False, slots)
        if isinstance(node, ast.Subscript):
            value = self.shape(rel, node.value, env, body, depth + 1)
            if (
                value.kind == "tuple"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)
            ):
                return value.item[node.slice.value] if 0 <= node.slice.value < len(value.item) else UNKNOWN
            if value.kind == "dict" and isinstance(node.slice, ast.Constant):
                return dict(value.slots).get(node.slice.value, value.item)
        if isinstance(node, (ast.Tuple, ast.List)):
            return Value("tuple", tuple(self.shape(rel, value, env, body, depth + 1) for value in node.elts))
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "dict"
                and node.func.id not in env
                and self.lookup(rel, node.func.id, body) is None
            ):
                return Value("dict", UNKNOWN)
            target = self.target(rel, node.func, body)
            if target and isinstance(target[1], ast.FunctionDef):
                fn = target[1]
                if fn.decorator_list or fn.args.vararg or fn.args.kwarg:
                    return UNKNOWN
                local = {
                    p.arg: self.shape(rel, value, env, body, depth + 1)
                    for p, value in zip(fn.args.posonlyargs + fn.args.args, node.args)
                }
                # Return-shape inference is separate from effect admission. It
                # never blesses a call receiving changed metadata as safe.
                statements = [
                    sub for sub in scope_statements(fn.body) if isinstance(sub, (ast.Assign, ast.Return))
                ]
                for statement in statements:
                    if isinstance(statement, ast.Assign):
                        value = self.shape(target[0], statement.value, local, fn.body, depth + 1)
                        for dest in statement.targets:
                            prior = local.copy()
                            bind(local, dest, value)
                            for name in set(prior) & set(local):
                                if local[name] != prior[name]:
                                    local[name] = merge([prior[name], local[name]])
                returns = [
                    self.shape(target[0], statement.value, local, fn.body, depth + 1)
                    for statement in statements
                    if isinstance(statement, ast.Return) and statement.value is not None
                ]
                return merge(returns)
        return UNKNOWN


def bind(env, node, value):
    if isinstance(node, ast.Name):
        env[node.id] = value
    elif isinstance(node, (ast.Tuple, ast.List)):
        items = value.item if value.kind == "tuple" else [replace(UNKNOWN, hot=value.hot)] * len(node.elts)
        for target, item in zip(node.elts, items):
            bind(env, target, item)


def with_hot(value):
    if value.kind == "dict":
        return Value(
            "dict", with_hot(value.item), True, tuple((key, with_hot(item)) for key, item in value.slots)
        )
    if value.kind == "tuple":
        return Value("tuple", tuple(with_hot(item) for item in value.item), True)
    return replace(value, hot=True)


class Effects:
    def __init__(self, source, rel, fn, env):
        self.source, self.rel, self.fn, self.env = source, rel, fn, env

    def derived(self, values):
        return Value("derived", hot=True) if any(v.hot for v in values) else SCALAR

    def expr(self, node):
        if node is None:
            return SCALAR
        if isinstance(node, ast.Name):
            return self.env.get(node.id, self.source.shape(self.rel, node, self.env, self.fn.body))
        if isinstance(node, ast.Constant):
            return SCALAR
        if isinstance(node, ast.Attribute):
            value = self.expr(node.value)
            if not value.hot:
                return UNKNOWN
            if value.kind == "class" and node.attr == "__mro__":
                return Value("iter", Value("class", hot=True), True)
            if value.kind == "class" and node.attr == "__dict__":
                return Value("class_dict", Value("descriptor", hot=True), True)
            if value.kind == "class" and node.attr == "__name__":
                return SCALAR
            raise Unsafe("unproved attribute descriptor on supplied metadata")
        if isinstance(node, ast.Subscript):
            value = self.expr(node.value)
            self.expr(node.slice)
            if value.kind == "class_dict":
                if not isinstance(node.slice, ast.Constant) or node.slice.value not in (
                    "__dict__",
                    "__del__",
                ):
                    raise Unsafe("non-layout class dictionary access")
                return Value("descriptor", hot=True)
            if value.kind == "dict":
                item = (
                    dict(value.slots).get(node.slice.value, value.item)
                    if isinstance(node.slice, ast.Constant)
                    else value.item
                )
                return replace(item, hot=value.hot or item.hot)
            if value.hot:
                raise Unsafe("unknown subscript protocol")
            return UNKNOWN
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            items = tuple(self.expr(v) for v in node.elts)
            return Value("tuple", items, any(v.hot for v in items))
        if isinstance(node, ast.Dict):
            keys = [self.expr(v) for v in node.keys if v is not None]
            values = [self.expr(v) for v in node.values]
            if any(k.hot and k.kind not in ("class", "scalar") for k in keys):
                raise Unsafe("dictionary key hook")
            slots = tuple(
                (key.value, value) for key, value in zip(node.keys, values) if isinstance(key, ast.Constant)
            )
            return Value("dict", merge(values), any(v.hot for v in keys + values), slots)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            saved = self.env.copy()
            try:
                for gen in node.generators:
                    if gen.is_async:
                        raise Unsafe("async comprehension")
                    bind(self.env, gen.target, self.iter_item(self.expr(gen.iter)))
                    for cond in gen.ifs:
                        if self.expr(cond).hot:
                            raise Unsafe("metadata-dependent comprehension membership")
                if isinstance(node, ast.DictComp):
                    key = self.expr(node.key)
                    value = self.expr(node.value)
                    if key.kind == "derived":
                        value = Value("derived", hot=True)
                    return Value("dict", value, value.hot)
                value = self.expr(node.elt)
                return Value("iter", value, value.hot)
            finally:
                self.env = saved
        if isinstance(node, ast.Call):
            args = [self.expr(v) for v in node.args] + [self.expr(k.value) for k in node.keywords]
            hot = any(v.hot for v in args)
            if isinstance(node.func, ast.Attribute):
                receiver = self.expr(node.func.value)
                name = node.func.attr
                if receiver.kind == "dict" and name in ("items", "values", "keys", "get", "pop", "update"):
                    if name == "items":
                        return Value(
                            "iter", Value("tuple", (SCALAR, receiver.item), receiver.hot), receiver.hot
                        )
                    if name == "values":
                        return Value("iter", receiver.item, receiver.hot)
                    if name == "keys":
                        return Value("iter", SCALAR)
                    if name in ("get", "pop"):
                        if args and args[0].hot and args[0].kind not in ("class", "scalar"):
                            raise Unsafe("mapping key protocol")
                        return replace(receiver.item, hot=receiver.hot or receiver.item.hot)
                    if any(value.hot and value.kind != "dict" for value in args[: len(node.args)]):
                        raise Unsafe("unknown mapping update protocol")
                    if any(
                        keyword.arg is None and value.hot and value.kind != "dict"
                        for keyword, value in zip(node.keywords, args[len(node.args) :])
                    ):
                        raise Unsafe("unknown keyword mapping protocol")
                    self.mark(node.func.value, merge(args))
                    return SCALAR
                if receiver.hot:
                    raise Unsafe("method called on supplied metadata")
                if hot and self.source.native_factory(self.rel, node.func, self.fn):
                    return Value("descriptor", hot=True)
            if isinstance(node.func, ast.Name):
                name = node.func.id
                function = self.expr(node.func)
                if function.hot:
                    raise Unsafe("supplied callable executed")
                builtin = name not in self.env and self.source.lookup(self.rel, name, self.fn.body) is None
                if builtin and name == "dict":
                    if hot and any(v.kind != "dict" for v in args):
                        raise Unsafe("unknown mapping protocol")
                    slots = tuple((key, value) for arg in args for key, value in arg.slots)
                    return Value("dict", merge(v.item for v in args if v.kind == "dict"), hot, slots)
                if builtin and name in ("tuple", "list", "set", "frozenset"):
                    value = self.iter_item(args[0]) if args else UNKNOWN
                    return Value("iter", value, value.hot)
                if builtin and name in ("any", "all", "next"):
                    value = self.iter_item(args[0]) if args else UNKNOWN
                    if name != "next" and value.hot and value.kind not in ("scalar", "class", "derived"):
                        raise Unsafe("truth hook")
                    return value if name == "next" else self.derived([value])
                if builtin and name in ("isinstance", "issubclass"):
                    if hot and any(
                        v.hot and v.kind not in ("class", "descriptor", "dict", "scalar") for v in args
                    ):
                        raise Unsafe("type-check hook")
                    if hot and (len(args) != 2 or args[1].kind not in ("class", "type_marker")):
                        raise Unsafe("type-check target hook")
                    # Called only for function edits: ordinary class ancestry
                    # and the native layout descriptor's type are unchanged.
                    return SCALAR
                if builtin and name in ("len", "type", "id"):
                    if hot and any(
                        v.kind not in ("class", "class_dict", "dict", "iter", "tuple", "descriptor", "scalar")
                        for v in args
                    ):
                        raise Unsafe("builtin protocol")
                    return self.derived(args)
                target = self.source.target(self.rel, node.func, self.fn.body)
                if target == ("types", "MethodType"):
                    return Value("bound", hot=hot)
            if hot:
                raise Unsafe("metadata escapes to an unknown consumer")
            return self.source.shape(self.rel, node, self.env, self.fn.body)
        if isinstance(node, ast.Compare):
            values = [self.expr(node.left)] + [self.expr(v) for v in node.comparators]
            if (
                len(node.ops) == 1
                and isinstance(node.ops[0], (ast.In, ast.NotIn))
                and isinstance(node.left, ast.Constant)
                and node.left.value in ("__dict__", "__del__")
                and values[1].kind == "class_dict"
            ):
                return SCALAR
            if any(v.hot and v.kind not in ("class", "dict", "scalar", "descriptor") for v in values):
                raise Unsafe("comparison hook")
            return self.derived(values)
        if isinstance(node, (ast.BoolOp, ast.UnaryOp, ast.IfExp)):
            values = [self.expr(v) for v in ast.iter_child_nodes(node) if isinstance(v, ast.expr)]
            if any(
                v.hot and v.kind not in ("class", "dict", "scalar", "descriptor", "derived") for v in values
            ):
                raise Unsafe("truth hook")
            return merge(values) if isinstance(node, ast.IfExp) else self.derived(values)
        values = [self.expr(v) for v in ast.iter_child_nodes(node) if isinstance(v, ast.expr)]
        if any(v.hot for v in values):
            raise Unsafe("unsupported metadata expression")
        return UNKNOWN

    def iter_item(self, value):
        if value.kind == "dict":
            return SCALAR
        if value.kind == "iter":
            return value.item
        if value.kind == "tuple":
            return merge(value.item)
        if value.hot:
            raise Unsafe("unknown iteration protocol")
        return UNKNOWN

    def mark(self, node, value):
        def derived(v):
            if v.kind == "derived":
                return True
            if v.kind in ("dict", "iter") and v.item is not None:
                return derived(v.item)
            if v.kind == "tuple":
                return any(derived(item) for item in v.item)
            return any(derived(item) for _key, item in v.slots)

        if derived(value):
            raise Unsafe("metadata-derived value is published")
        if isinstance(node, ast.Name):
            old = self.env.get(node.id, UNKNOWN)
            slots = dict(old.slots)
            slots.update(dict(value.slots))
            self.env[node.id] = Value(
                "dict", merge([old.item or UNKNOWN, value]), old.hot or value.hot, tuple(slots.items())
            )
        elif isinstance(node, ast.Subscript):
            parent = self.expr(node.value)
            if value.hot and parent.kind != "dict":
                raise Unsafe("unknown mapping assignment hook")
            slots = dict(parent.slots)
            if isinstance(node.slice, ast.Constant):
                slots[node.slice.value] = value
            changed = Value(
                "dict", merge([parent.item or UNKNOWN, value]), parent.hot or value.hot, tuple(slots.items())
            )
            if isinstance(node.value, ast.Name):
                self.env[node.value.id] = changed
            else:
                self.mark(node.value, changed)
        elif isinstance(node, ast.Attribute):
            parent = self.expr(node.value)
            if (value.hot or parent.hot) and parent.kind != "class":
                raise Unsafe("attribute assignment hook")
        elif value.hot:
            raise Unsafe("unknown metadata mutation")

    def statements(self, body):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass, ast.Continue, ast.Break)):
                continue
            if isinstance(node, ast.Assign):
                value = self.expr(node.value)
                for dest in node.targets:
                    if isinstance(dest, (ast.Name, ast.Tuple, ast.List)):
                        bind(self.env, dest, value)
                    else:
                        self.mark(dest, value)
            elif isinstance(node, ast.Expr):
                self.expr(node.value)
            elif isinstance(node, (ast.If, ast.For)):
                if isinstance(node, ast.For):
                    bind(self.env, node.target, self.iter_item(self.expr(node.iter)))
                condition = self.expr(node.test) if isinstance(node, ast.If) else SCALAR
                if condition.hot:
                    raise Unsafe("metadata-dependent control flow")
                before = self.env.copy()
                self.statements(node.body)
                first = self.env.copy()
                self.env = before.copy()
                self.statements(node.orelse)
                self.env = {
                    name: merge([first.get(name, UNKNOWN), self.env.get(name, UNKNOWN)])
                    for name in set(first) | set(self.env)
                }
            elif isinstance(node, ast.Return):
                if self.expr(node.value).hot:
                    raise Unsafe("metadata returned to unknown consumer")
            elif isinstance(node, ast.Raise):
                self.expr(node.exc)
            else:
                raise Unsafe("unsupported connector statement")


def metadata_only_call(rel, call, affected, parse, native_factory, registry_guard=None):
    source = Source(parse, native_factory, registry_guard, (rel, call.lineno))
    target = source.target(rel, call.func)
    if not target or not isinstance(target[1], ast.FunctionDef):
        return False
    fn = target[1]
    if fn.decorator_list or fn.args.vararg or fn.args.kwarg:
        return False
    params = fn.args.posonlyargs + fn.args.args
    env = {p.arg: UNKNOWN for p in params + fn.args.kwonlyargs}
    values = list(zip((p.arg for p in params), call.args))
    values += [(k.arg, k.value) for k in call.keywords]
    for name, node in values:
        if name is None or name not in env:
            return False
        value = source.shape(rel, node)
        if any(isinstance(n, ast.Name) and n.id in affected for n in ast.walk(node)):
            value = with_hot(value)
        env[name] = value
    try:
        Effects(source, target[0], fn, env).statements(fn.body)
    except (Unsafe, RecursionError, TypeError, AttributeError):
        return False
    return True
