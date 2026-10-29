"""Source witnesses for mode and CPU binding routes; never load product code."""
import ast
from collections import Counter
import hashlib
from pathlib import Path


def _shape(node):
    return ast.dump(node, include_attributes=False)


def _expression(text):
    return _shape(ast.parse(text, mode="eval").body)


def _contains(node, expression):
    shape = _expression(expression)
    return any(_shape(child) == shape for child in ast.walk(node))


def _body_matches(node, expected):
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return [_shape(item) for item in body] == [_shape(item) for item in ast.parse(expected).body]


CPU_SELECTOR = '''
global _SELECTED, _CPU_ONLY
for name in _MODULES:
    full = f"{pkg.__name__}.{name}"
    basename = _HOST_MODULES.get(name)
    if not (basename and os.path.exists(host_module_path(basename))):
        fallback = _HOST_INFERENCE_MODULES.get(name)
        basename = fallback if fallback and os.path.exists(host_module_path(fallback)) else None
    if basename:
        module = _HostBinding(full, name, basename, reason)
    else:
        module = _NoGpuBinding(full, reason)
        if name not in _MISSING:
            _MISSING.append(name)
    sys.modules[full] = module
    setattr(pkg, name, module)
_CPU_ONLY = reason
_SELECTED = mode
return _SELECTED
'''

DEFAULT_MODE = '''
global _DEFAULT_MODE
if _DEFAULT_MODE is None:
    _DEFAULT_MODE = _SELECTED or requested_mode()
return _DEFAULT_MODE
'''



def optional_file_route(tree, function, constants):
    """A directly tested native file may be absent from a selected install.

    This proves only the load's availability predicate, not mode exclusivity.
    The requirement must also bind a selected layout and runtime absence.
    """
    imports_os = any(isinstance(n, ast.Import) and any(a.name == 'os' and a.asname in (None, 'os') for a in n.names) for n in tree.body)
    def root_name(node):
        while isinstance(node, ast.Attribute):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None
    mutations = any((isinstance(n, (ast.Name, ast.Attribute)) and isinstance(n.ctx, ast.Store)
                     and root_name(n) in ('os', '_backend'))
                    or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'setattr'
                        and n.args and root_name(n.args[0]) in ('os', '_backend')) for n in ast.walk(tree))
    if not imports_os or mutations:
        return None
    for index, guard in enumerate(function.body):
        if not (isinstance(guard, ast.If) and not guard.orelse and len(guard.body) == 1
                and isinstance(guard.body[0], ast.Return) and isinstance(guard.test, ast.UnaryOp)
                and isinstance(guard.test.op, ast.Not) and isinstance(guard.test.operand, ast.Call)):
            continue
        returned = guard.body[0].value
        if isinstance(returned, ast.Name):
            values = [n.value for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == returned.id for t in n.targets)]
            if len(values) != 1:
                continue
            returned = values[0]
        try:
            ast.literal_eval(returned)
        except (ValueError, TypeError):
            continue
        exists = guard.test.operand
        if (_shape(exists.func) != _expression('os.path.exists') or len(exists.args) != 1
                or exists.keywords or not isinstance(exists.args[0], ast.Name)):
            continue
        path = exists.args[0].id
        assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                       and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == path]
        if len(assignments) != 1 or sum(isinstance(n, ast.Name) and n.id == path and isinstance(n.ctx, ast.Store) for n in ast.walk(function)) != 1:
            continue
        assignment = assignments[0]
        before = {id(n) for stmt in function.body[:index] for n in ast.walk(stmt)}
        if id(assignment) not in before:
            continue
        for call in (n for stmt in function.body[index+1:] for n in ast.walk(stmt) if isinstance(n, ast.Call)):
            if (_shape(call.func) != _expression('_backend._load_member') or len(call.args) != 3 or call.keywords
                    or not isinstance(call.args[0], ast.Constant) or call.args[0].value not in ('fast', 'identical')
                    or not isinstance(call.args[1], ast.Name) or _shape(call.args[2]) != _expression(path)):
                continue
            tier, symbol = call.args[0].value, call.args[1].id
            binding = constants.get(symbol)
            if not binding or not binding.startswith('_mojolearn'):
                continue
            expected = f'os.path.join(_backend.tier_dir({tier!r}), {symbol} + ".so")'
            if _shape(assignment.value) != _expression(expected):
                continue
            # An earlier/unconditional load invalidates this optionality proof.
            uses = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                    and _shape(n.func) == _expression('_backend._load_member')
                    and any(isinstance(a, ast.Name) and a.id == symbol for a in n.args)]
            symbol_reads = sum(isinstance(n, ast.Name) and n.id == symbol and isinstance(n.ctx, ast.Load) for n in ast.walk(function))
            if len(uses) != 1 or symbol_reads != 2:
                continue
            return dict(binding=binding, symbol=symbol, tier=tier, basename=binding+'.so',
                        function=function.name, availability=ast.unparse(guard.test),
                        function_sha256=hashlib.sha256(_shape(function).encode()).hexdigest())
    return None


def loader_routes(root, surface):
    """Read the same frozen metadata registry already used by lane_select.

    Registry aliases count only when the frozen loader actually consumes that
    map. Unknown loader syntax or unguarded binding uses stay unresolved.
    """
    root = Path(root)
    sources, aliases, excluded, optional = {}, {}, {}, {}

    def read(path):
        data = (root / path).read_bytes()
        sources[path] = hashlib.sha256(data).hexdigest()
        return ast.parse(data, filename=path)

    backend_path = "python/mojolearn/_backend.py"
    surface_path = "python/mojolearn/host_surface.py"
    if not (root / backend_path).is_file() or not (root / surface_path).is_file():
        return dict(sources=sources, cpu_aliases=aliases, mode_exclusions=excluded, optional_files=optional,
                    unresolved=["frozen loader/host registry source unavailable"])
    backend, registry = read(backend_path), read(surface_path)
    assignments = {target.id: node.value for node in backend.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    functions = {node.name: node for node in backend.body if isinstance(node, ast.FunctionDef)}
    registry_functions = {node.name: node for node in registry.body if isinstance(node, ast.FunctionDef)}
    select = functions.get("_select_cpu_only")
    direct_map = assignments.get("_HOST_MODULES")
    routes_function = registry_functions.get("routed_modules")
    direct = (direct_map is not None and _shape(direct_map) == _expression("host_surface.routed_modules()")
        and select is not None and _body_matches(select, CPU_SELECTOR)
        and routes_function is not None
        and _contains(routes_function, '{f["routes"]: f["binding"] for f in FAMILIES if f["routes"]}'))
    if direct:
        for logical, actual in surface.routed_modules().items():
            aliases[logical] = dict(bindings=[actual], route="_HOST_MODULES", sources=[backend_path, surface_path])
    # Expansion families are loaded by the registry itself, so retain each
    # fragment as evidence for the values returned by routed_modules().
    for path in sorted((root / "python/mojolearn").glob("_surface_*.py")):
        read(str(path.relative_to(root)))
    fallback = assignments.get("_HOST_INFERENCE_MODULES")
    if (direct and fallback is not None and _shape(fallback) == _expression("host_surface.inference_routes()")
            and _contains(select, "_HOST_INFERENCE_MODULES.get(name)")):
        for logical, actual in surface.inference_routes().items():
            row = aliases.setdefault(logical, dict(bindings=[], route="_HOST_INFERENCE_MODULES", sources=[backend_path, surface_path]))
            if actual not in row["bindings"]:
                row["bindings"].append(actual)
    cpu = functions.get("_cpu_only_binding")
    if (cpu is not None and _contains(cpu, "host_surface.ADAPTED_MODULES.get(name)")
            and _contains(cpu, 'host_surface.family(adapted["family"])["binding"]')):
        for logical, adapted in surface.ADAPTED_MODULES.items():
            actual = surface.family(adapted["family"])["binding"]
            row = aliases.setdefault(logical, dict(bindings=[], route="ADAPTED_MODULES", sources=[backend_path, surface_path]))
            if actual not in row["bindings"]:
                row["bindings"].append(actual)

    default_mode = functions.get("default_mode")
    known_default = default_mode is not None and _body_matches(default_mode, DEFAULT_MODE)
    parsed = [(path, ast.parse(path.read_bytes(), filename=str(path)))
              for path in sorted((root / "python/mojolearn").rglob("*.py"))]
    literals = Counter(node.value for _, tree in parsed for node in ast.walk(tree)
                       if isinstance(node, ast.Constant) and isinstance(node.value, str)
                       and node.value.startswith("_mojolearn"))
    for path, tree in parsed:
        backend_import = any(isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
            and any(alias.name == "_backend" and alias.asname in (None, "_backend") for alias in node.names)
            for node in tree.body)
        if not known_default or not backend_import or any(isinstance(node, ast.Name) and node.id == "_backend"
                and isinstance(node.ctx, ast.Store) for node in ast.walk(tree)):
            continue
        constants = {target.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
                     and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                     for target in node.targets if isinstance(target, ast.Name)}
        for function in (node for node in tree.body if isinstance(node, ast.FunctionDef) and not node.decorator_list):
            candidate = optional_file_route(tree, function, constants)
            if candidate and literals[candidate['binding']] == 1:
                symbol = candidate['symbol']
                descendants = {id(node) for node in ast.walk(function)}
                outside = any(id(node) not in descendants for node in ast.walk(tree)
                    if isinstance(node, ast.Name) and node.id == symbol and isinstance(node.ctx, ast.Load))
                external = any((isinstance(node, ast.ImportFrom) and any(alias.name == symbol or (alias.name == '*' and (node.module or '').split('.')[-1] == path.stem) for alias in node.names))
                    or (isinstance(node, ast.Attribute) and node.attr == symbol)
                    or (isinstance(node, ast.Constant) and node.value == symbol)
                    for other_path, other in parsed if other_path != path for node in ast.walk(other))
                stores = sum(isinstance(node, ast.Name) and node.id == symbol and isinstance(node.ctx, ast.Store) for node in ast.walk(tree))
                if not outside and not external and stores == 1:
                    rel = str(path.relative_to(root))
                    sources[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
                    optional[candidate['binding']] = dict(candidate, source=rel)
            body = function.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            if not body or not isinstance(body[0], ast.If):
                continue
            guard = body[0]
            test = guard.test
            if (guard.orelse or len(guard.body) != 1 or not isinstance(guard.body[0], ast.Return)
                    or not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.NotEq)
                    or len(test.comparators) != 1 or not isinstance(test.comparators[0], ast.Constant)
                    or test.comparators[0].value not in ("fast", "identical")
                    or _shape(test.left) != _expression("_backend.default_mode()")):
                continue
            mode = test.comparators[0].value
            descendants = {id(node) for node in ast.walk(function)}
            for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
                if (_shape(call.func) != _expression("_backend._load_member") or len(call.args) < 2
                        or not isinstance(call.args[0], ast.Constant) or call.args[0].value != mode
                        or not isinstance(call.args[1], ast.Name)):
                    continue
                symbol = call.args[1].id
                binding = constants.get(symbol)
                if not binding or not binding.startswith("_mojolearn") or literals[binding] != 1:
                    continue
                # A local guard cannot exclude another module's direct or
                # imported use of the same binding. Unknown external use is
                # retained as a missing route, never assumed unreachable.
                if any((isinstance(node, ast.ImportFrom) and any(alias.name == symbol or
                        (alias.name == "*" and (node.module or "").split(".")[-1] == path.stem)
                        for alias in node.names))
                        or (isinstance(node, ast.Attribute) and node.attr == symbol)
                        or (isinstance(node, ast.Constant) and node.value == symbol)
                        for other_path, other in parsed if other_path != path for node in ast.walk(other)):
                    continue
                if sum(isinstance(node, ast.Name) and node.id == symbol and isinstance(node.ctx, ast.Store)
                       for node in ast.walk(tree)) != 1:
                    continue
                if any(id(node) not in descendants for node in ast.walk(tree)
                       if isinstance(node, ast.Name) and node.id == symbol and isinstance(node.ctx, ast.Load)):
                    continue
                rel = str(path.relative_to(root))
                sources[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
                excluded.setdefault("identical" if mode == "fast" else "fast", {})[binding] = dict(
                    active_mode=mode, source=rel, function=function.name, guard=ast.unparse(test),
                    declared_default_mode="identical" if mode == "fast" else "fast",
                    function_sha256=hashlib.sha256(_shape(function).encode()).hexdigest())
    return dict(sources=sources, cpu_aliases=aliases, mode_exclusions=excluded, optional_files=optional,
                unresolved=[] if direct else ["frozen CPU loader route could not be proven"])
