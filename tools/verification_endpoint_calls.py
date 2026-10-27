"""Direct public calls in identity fixtures; no product imports or execution.

Keep receiver types: seeing ``fit`` and ``KMeans`` somewhere in one file is
not evidence that a fixture calls KMeans.fit. Unknown receivers stay unknown.
These are source mappings, not executed endpoint receipts.
"""
import ast
import inspect
import textwrap
from collections import defaultdict


class Calls:
    def __init__(self, endpoints, helpers=None):
        self.endpoints = {}
        self.classes = set()
        for row in endpoints:
            for alias in row['aliases']:
                self.endpoints[alias + '.' + row['op']] = row['endpoint']
                if row['kind'] != 'function':
                    self.classes.add(alias)
        self.helpers = helpers or {}
        self.seen, self.active = set(), set()

    def function(self, node, env=None):
        env = dict(env or {})
        returns = set()
        helpers = dict(self.helpers)
        helpers.update({n.name: n for n in node.body if isinstance(n, ast.FunctionDef)})

        def returned(path):
            if path in ('mojolearn.ops.Session.capture', 'mojolearn.ops.Session.load_program'):
                return {'mojolearn.ops.Program'}
            if path.endswith('.assert_native') and path.startswith('@audit:'):
                return {path[len('@audit:'):-len('.assert_native')]}
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
                return paths
            if isinstance(expr, ast.Call):
                args = [value(a) for a in expr.args]
                for kw in expr.keywords:
                    value(kw.value)
                paths, result = value(expr.func), set()
                # The graph verifier's sole invocation helper delegates to
                # audit_native_call(operation, ...), which executes its first
                # argument. Passing a method to an arbitrary helper is NOT
                # enough to establish this call.
                if 'mojolearn._verify_native_runtime.audit_native_call' in paths and args:
                    for operation in args[0]:
                        record(operation)
                        result.update('@audit:' + p for p in returned(operation))
                for path in paths:
                    if path in self.classes:
                        record(path + '.__init__')
                        result.add(path)
                    else:
                        record(path)
                        record(path + '.call')
                    owner, _, method = path.rpartition('.')
                    if owner in self.classes and method in ('fit', 'partial_fit', 'set_params'):
                        result.add(owner)
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
                return value(expr.value)
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
                        bind(target, values)
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
