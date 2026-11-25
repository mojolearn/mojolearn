"""Direct public calls in identity fixtures; no product imports or execution.

Keep receiver types: seeing ``fit`` and ``KMeans`` somewhere in one file is
not evidence that a fixture calls KMeans.fit. Unknown receivers stay unknown.
These are source mappings, not executed endpoint receipts.
"""
import ast
import inspect
import textwrap
from collections import defaultdict

#: Classmethod loaders whose result is an instance of their own class.
LOADERS = frozenset(('load', 'from_checkpoint', 'from_checkpoint_dir', 'from_checkpoint_file', 'from_pretrained',
                     'from_dict', 'from_list'))

#: Public functions whose result is an instance of a public class, by the
#: class's short name (its aliases are read from the endpoint rows).
RETURNS_CLASS = {'mojolearn.importers.import_model': 'ImportedProgram'}

#: Verification helpers that EXECUTE one of their arguments, by the position
#: of the operation they run: `audit_native_call(operation, ...)` and the
#: program verifiers' shared `audited(audits, name, operation, ...)` /
#: `audited_context(audits, name, operation, ...)`
#: (python/mojolearn/_verify_native_runtime.py). Passing a method to any other
#: helper is NOT a call of it.
EXECUTORS = {'mojolearn._verify_native_runtime.audit_native_call': 0,
             'mojolearn._verify_native_runtime.audited': 2,
             'mojolearn._verify_native_runtime.audited_context': 2}

#: `functools.partial(f, *bound)` keeps `f` and the number of positional
#: arguments it binds, so a later call of the partial finds the operation an
#: executor runs at its shifted position.
PARTIAL = '@partial:'


class Instance(str):
    """A path that names an INSTANCE of a public class, not the class: a call
    on it is ``__call__``, ``in`` is ``__contains__`` and ``[]`` is
    ``__getitem__``. Attribute paths built from it are plain strings, so
    ``obj.method`` still reads ``<Class>.method``."""
    __slots__ = ()


class Calls:
    def __init__(self, endpoints, helpers=None):
        self.endpoints = {}
        self.classes = set()
        # Attributes installed on the class after its body (`Array.flags =
        # property(...)`): READING one runs its descriptor, so an attribute
        # read reaches the endpoint as the class's own `<name>:get` does.
        self.installed = {}
        for row in endpoints:
            for alias in row['aliases']:
                self.endpoints[alias + '.' + row['op']] = row['endpoint']
                if row['kind'] == 'installed_method':
                    self.installed[alias + '.' + row['op']] = row['endpoint']
                if row['kind'] != 'function':
                    self.classes.add(alias)
        self.helpers = helpers or {}
        self.seen, self.active = set(), set()

    def _instances(self, short):
        return {Instance(alias) for alias in self.classes if alias.rpartition('.')[2] == short}

    def function(self, node, env=None):
        env = dict(env or {})
        returns = set()
        helpers = dict(self.helpers)
        helpers.update({n.name: n for n in node.body if isinstance(n, ast.FunctionDef)})

        def returned(path):
            if path in ('mojolearn.ops.Session.capture', 'mojolearn.ops.Session.load_program'):
                return {Instance('mojolearn.ops.Program')}
            if path.endswith('.assert_native') and path.startswith('@audit:'):
                return {path[len('@audit:'):-len('.assert_native')]}
            if path in RETURNS_CLASS:
                return self._instances(RETURNS_CLASS[path])
            owner, _, method = path.rpartition('.')
            if owner in self.classes and (method in LOADERS or method.startswith('from_checkpoint')):
                return {Instance(owner)}
            return set()

        def record(path):
            endpoint = self.endpoints.get(path)
            if endpoint:
                self.seen.add(endpoint)

        def bind(target, values):
            if isinstance(target, ast.Name):
                env[target.id] = set(values)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for child in target.elts:
                    bind(child, values)

        def value(expr):
            if expr is None:
                return set()
            if isinstance(expr, ast.Name):
                return env.get(expr.id, set())
            if isinstance(expr, ast.Attribute):
                paths = {p + '.' + expr.attr for p in value(expr.value)}
                for path in paths:
                    record(path + ':get')
                    if path in self.installed:
                        self.seen.add(self.installed[path])
                return paths
            if isinstance(expr, ast.IfExp):
                # `ops if side is None else vd.view(ops, side)`: either branch
                # is what the expression can hold
                value(expr.test)
                return value(expr.body) | value(expr.orelse)
            if isinstance(expr, ast.Call):
                args = [value(a) for a in expr.args]
                for kw in expr.keywords:
                    value(kw.value)
                paths, result = value(expr.func), set()
                # An executor (EXECUTORS) runs the argument at its operation
                # position, directly or through a partial that bound some of
                # its leading arguments. Passing a method to an arbitrary
                # helper is NOT enough to establish this call.
                for path in paths:
                    bound, target = 0, path
                    if path.startswith(PARTIAL):
                        count, _, target = path[len(PARTIAL):].partition(':')
                        bound = int(count)
                    position = EXECUTORS.get(target)
                    if position is None or not 0 <= position - bound < len(args):
                        continue
                    for operation in args[position - bound]:
                        record(operation)
                        if target.endswith('.audit_native_call'):
                            result.update('@audit:' + p for p in returned(operation))
                        else:
                            result.update(returned(operation))
                if 'functools.partial' in paths and args:
                    result.update(f'{PARTIAL}{len(args) - 1}:{p}' for p in args[0] if not p.startswith(PARTIAL))
                if isinstance(expr.func, ast.Name) and expr.func.id not in env and args:
                    # builtins that run a dunder of their first argument
                    builtin = expr.func.id
                    for path in args[0]:
                        if not isinstance(path, Instance):
                            continue
                        if builtin == 'hash':
                            record(path + '.__hash__')
                        elif builtin in ('delattr', 'setattr'):
                            record(path + ('.__delattr__' if builtin == 'delattr' else '.__setattr__'))
                            name = expr.args[1] if len(expr.args) > 1 else None
                            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                                record(path + '.' + name.value + (':delete' if builtin == 'delattr' else ':set'))
                for path in paths:
                    if isinstance(path, Instance):
                        # `obj(x)` on an instance runs the class's __call__
                        record(path + '.__call__')
                    elif path in self.classes:
                        record(path + '.__init__')
                        result.add(Instance(path))
                    else:
                        record(path)
                        record(path + '.call')
                    owner, _, method = path.rpartition('.')
                    if owner in self.classes and method in ('fit', 'partial_fit', 'set_params'):
                        result.add(Instance(owner))
                    result.update(returned(path))
                if isinstance(expr.func, ast.Name) and expr.func.id in helpers:
                    key = expr.func.id, tuple(tuple(sorted(a)) for a in args)
                    if key not in self.active:
                        helper = helpers[expr.func.id]
                        self.active.add(key)
                        bound = {p.arg: a for p, a in zip(helper.args.args, args)}
                        previous = self.helpers
                        self.helpers = helpers
                        try:
                            result |= self.function(helper, {**env, **bound})
                        finally:
                            self.helpers = previous
                        self.active.remove(key)
                return result
            if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
                return set().union(*(value(e) for e in expr.elts)) if expr.elts else set()
            if isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for generator in expr.generators:
                    bind(generator.target, value(generator.iter))
                    for condition in generator.ifs:
                        value(condition)
                if isinstance(expr, ast.DictComp):
                    return value(expr.key) | value(expr.value)
                return value(expr.elt)
            if isinstance(expr, ast.Subscript):
                value(expr.slice)
                held = value(expr.value)
                for path in held:
                    if isinstance(path, Instance):
                        record(path + '.__getitem__')
                return held
            if isinstance(expr, ast.Compare):
                # `key in obj` / `key not in obj` runs the right operand's
                # __contains__
                # `a == b` / `a != b` runs the left operand's __eq__ / __ne__
                left = value(expr.left)
                for op, comparator in zip(expr.ops, expr.comparators):
                    held = value(comparator)
                    if isinstance(op, (ast.In, ast.NotIn)):
                        for path in held:
                            if isinstance(path, Instance):
                                record(path + '.__contains__')
                    elif isinstance(op, (ast.Eq, ast.NotEq)):
                        for path in left:
                            if isinstance(path, Instance):
                                record(path + ('.__eq__' if isinstance(op, ast.Eq) else '.__ne__'))
                    left = held
                return set()
            if isinstance(expr, (ast.Yield, ast.YieldFrom)):
                yielded = value(expr.value)
                returns.update(yielded)
                return yielded
            # Comprehensions may contain calls with a typed receiver.
            for child in ast.iter_child_nodes(expr):
                if isinstance(child, ast.expr):
                    value(child)
            return set()

        def statements(nodes):
            for n in nodes:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(n, ast.ImportFrom):
                    module = n.module or ''
                    if n.level:
                        module = 'mojolearn' + ('.' + module if module else '')
                    for alias in n.names:
                        env[alias.asname or alias.name] = {module + '.' + alias.name}
                elif isinstance(n, ast.Import):
                    for alias in n.names:
                        env[alias.asname or alias.name.split('.')[0]] = {alias.name if alias.asname else alias.name.split('.')[0]}
                elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                    values = value(n.value)
                    for target in n.targets if isinstance(n, ast.Assign) else [n.target]:
                        if isinstance(target, ast.Attribute):
                            # `obj.name = v` runs the class's __setattr__ and the
                            # property's setter
                            for path in value(target.value):
                                if isinstance(path, Instance):
                                    record(path + '.__setattr__')
                                    record(path + '.' + target.attr + ':set')
                        else:
                            bind(target, values)
                elif isinstance(n, ast.Delete):
                    for target in n.targets:
                        if isinstance(target, ast.Attribute):
                            # `del obj.name` runs __delattr__ and the property's deleter
                            for path in value(target.value):
                                if isinstance(path, Instance):
                                    record(path + '.__delattr__')
                                    record(path + '.' + target.attr + ':delete')
                        else:
                            value(target)
                elif isinstance(n, (ast.With, ast.AsyncWith)):
                    for item in n.items:
                        bind(item.optional_vars, value(item.context_expr))
                    statements(n.body)
                elif isinstance(n, (ast.For, ast.AsyncFor)):
                    bind(n.target, value(n.iter))
                    statements(n.body)
                    statements(n.orelse)
                elif isinstance(n, ast.Return):
                    returns.update(value(n.value))
                elif isinstance(n, (ast.If, ast.While)):
                    value(n.test)
                    statements(n.body)
                    statements(n.orelse)
                elif isinstance(n, ast.Try):
                    for body in [n.body, n.orelse, n.finalbody, *[h.body for h in n.handlers]]:
                        statements(body)
                else:
                    for child in ast.iter_child_nodes(n):
                        if isinstance(child, ast.expr):
                            value(child)
        statements(node.body)
        return returns


def source_calls(functions, endpoints, roots):
    """Endpoint -> named fixture list, with helper calls resolved by arguments."""
    result = defaultdict(set)
    for name in roots:
        if name not in functions:
            continue
        visitor = Calls(endpoints, functions)
        visitor.function(functions[name], {'ml': {'mojolearn'}, 'mojolearn': {'mojolearn'}})
        for endpoint in visitor.seen:
            result[endpoint].add(name)
    return result


def lane_calls(harness, endpoints):
    helpers = {}
    for name, fn in vars(harness).items():
        if inspect.isfunction(fn):
            try:
                helpers[name] = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
            except (OSError, TypeError, IndentationError, SyntaxError):
                pass
    result = defaultdict(set)
    for lane, fn in harness.LANES.items():
        try:
            node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        except (OSError, TypeError, IndentationError, SyntaxError):
            continue
        visitor = Calls(endpoints, helpers)
        visitor.function(node, {'ml': {'mojolearn'}, 'mojolearn': {'mojolearn'}})
        for endpoint in visitor.seen:
            result[endpoint].add(lane)
    return result
