"""Source witnesses for mode and CPU binding routes; never load product code."""
import ast
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


def loader_routes(root, surface):
    """Read the same frozen metadata registry already used by lane_select.

    Registry aliases count only when the frozen loader actually consumes that
    map. Unknown loader syntax or unguarded binding uses stay unresolved.
    """
    root = Path(root)
    sources, aliases, excluded = {}, {}, {}

    def read(path):
        data = (root / path).read_bytes()
        sources[path] = hashlib.sha256(data).hexdigest()
        return ast.parse(data, filename=path)

    backend_path = "python/mojolearn/_backend.py"
    surface_path = "python/mojolearn/host_surface.py"
    if not (root / backend_path).is_file() or not (root / surface_path).is_file():
        return dict(sources=sources, cpu_aliases=aliases, mode_exclusions=excluded,
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
    for path in sorted((root / "python/mojolearn").rglob("*.py")):
        tree = ast.parse(path.read_bytes(), filename=str(path))
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
                if not binding or not binding.startswith("_mojolearn"):
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
    return dict(sources=sources, cpu_aliases=aliases, mode_exclusions=excluded,
                unresolved=[] if direct else ["frozen CPU loader route could not be proven"])
