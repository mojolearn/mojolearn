"""Conservative input proof for unchanged whole test files across test revisions.

Explicit-file pytest selection does not execute every other test module. Prove
the selected files and statically resolved test helpers unchanged. Refuse dynamic
imports/code loading that cannot be resolved, and retain every reason. This is
source-input equivalence, never a successful runtime observation.
"""
import ast
import hashlib
from pathlib import Path

SCHEMA = "mojolearn.runtime-test-tree-reuse.v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prove(before, after_root, selected, after=None):
    root = Path(after_root).resolve()
    actual = {p.relative_to(root).as_posix(): digest(p)
              for p in root.rglob("*") if p.is_file() and not p.is_symlink()}
    supplied_drift = after is not None and after != actual
    after = actual
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    result = dict(schema=SCHEMA, status="proved", selected_files=list(selected),
                  changed_files=changed, closure={}, unresolved=[])
    if supplied_drift or any(p.is_symlink() for p in root.rglob("*")):
        result["unresolved"].append(dict(reason="frozen tree hashes differ or contain a symlink"))
    if not selected or any(f not in before or f not in after for f in selected):
        result["unresolved"].append(dict(reason="selected file absent from a frozen tree"))
    if not changed:
        result["closure"] = dict(after)
        result["method"] = "entire-frozen-tree-byte-equality"
        result["status"] = "blocked" if result["unresolved"] else "proved"
        return result
    result["method"] = "selected-files-static-helper-closure-and-data-equality"
    for name in changed:
        if not name.endswith(".py"):
            result["unresolved"].append(dict(file=name, reason="test data asset changed"))
    pending, seen = list(selected), set()

    def refuse(file, node, reason):
        row = dict(file=file, line=getattr(node, "lineno", None), reason=reason)
        if row not in result["unresolved"]:
            result["unresolved"].append(row)

    def ancestors(name):
        for parent in (Path(name).parent, *Path(name).parent.parents):
            if str(parent) == "." or not parent.is_absolute():
                for basename in ("__init__.py", "conftest.py"):
                    rel = (parent / basename).as_posix()
                    if rel in before or rel in after:
                        pending.append(rel)

    def local_module(module, file, node):
        if module == "mojolearn.tests":
            relative = ""
        elif module.startswith("mojolearn.tests."):
            relative = module[len("mojolearn.tests."):].replace(".", "/")
        elif module.startswith("tests."):
            relative = module[len("tests."):].replace(".", "/")
        else:
            relative = module.replace(".", "/")
            if not any(p in before or p in after for p in
                       (relative + ".py", relative + "/__init__.py")):
                return
        choices = [relative + "/__init__.py", relative + ".py"] if relative else ["__init__.py"]
        old = next((p for p in choices if p in before), None)
        new = next((p for p in choices if p in after), None)
        if old != new:
            refuse(file, node, "test import resolution changed: " + module)
        if old is not None:
            pending.append(old)
        elif relative:
            # Namespace packages may have no __init__; named from-imports are
            # resolved separately. An absent explicit module is not invented.
            prefix = relative + "/"
            if not any(p.startswith(prefix) for p in set(before) | set(after)):
                refuse(file, node, "unresolved test import: " + module)

    while pending:
        file = pending.pop()
        if file in seen:
            continue
        seen.add(file)
        path = root / file
        if (Path(file).is_absolute() or ".." in Path(file).parts or not path.is_file()
                or path.is_symlink() or not path.resolve().is_relative_to(root)
                or before.get(file) != after.get(file) or digest(path) != after.get(file)):
            refuse(file, None, "selected file or possible helper bytes changed")
            continue
        result["closure"][file] = after[file]
        ancestors(file)
        if not file.endswith(".py"):
            continue
        try:
            tree = ast.parse(path.read_text(), filename=file)
        except (SyntaxError, UnicodeError) as error:
            refuse(file, None, "test source cannot be analyzed: " + type(error).__name__)
            continue
        aliases, assignments = {}, {}
        package = ["mojolearn", "tests", *Path(file).parent.parts]
        if package[-1:] == ["."]:
            package.pop()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
                    local_module(alias.name, file, node)
            elif isinstance(node, ast.ImportFrom):
                base = ".".join(package[:len(package) - node.level + 1]) if node.level else ""
                module = ".".join(x for x in (base, node.module or "") if x)
                local_module(module, file, node)
                for alias in node.names:
                    name = module + "." + alias.name
                    aliases[alias.asname or alias.name] = name
                    # from a module import a symbol is not necessarily a module.
                    relative = name.removeprefix("mojolearn.tests.").replace(".", "/")
                    if any(p in before or p in after for p in (relative + ".py", relative + "/__init__.py")):
                        local_module(name, file, node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(node.value)

        def qualified(node, trail=()):
            if isinstance(node, ast.Name):
                if node.id in aliases:
                    return aliases[node.id]
                values = assignments.get(node.id, [])
                if len(values) == 1 and node.id not in trail:
                    return qualified(values[0], (*trail, node.id)) or node.id
                return node.id
            if isinstance(node, ast.Attribute):
                base = qualified(node.value, trail)
                return base + "." + node.attr if base else node.attr
            return ""

        def literal(node, trail=()):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name) and node.id not in trail:
                values = assignments.get(node.id, [])
                resolved = [literal(v, (*trail, node.id)) for v in values]
                if resolved and all(x == resolved[0] for x in resolved):
                    return resolved[0]
            if isinstance(node, (ast.List, ast.Tuple)):
                return [literal(x, trail) for x in node.elts]
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left, right = literal(node.left, trail), literal(node.right, trail)
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
            return None

        for value in assignments.get("pytest_plugins", []):
            plugins = literal(value)
            if isinstance(plugins, str):
                plugins = [plugins]
            if not isinstance(plugins, list) or not all(isinstance(x, str) for x in plugins):
                refuse(file, value, "dynamic pytest plugin imports")
            else:
                for plugin in plugins:
                    local_module(plugin, file, value)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "__dict__" and qualified(node.value) in ("importlib", "builtins"):
                refuse(file, node, "reflective import namespace access")
            if not isinstance(node, ast.Call):
                continue
            name = qualified(node.func)
            tail = name.rsplit(".", 1)[-1]
            if tail in ("__import__", "import_module", "find_spec"):
                if tail == "__import__" and (len(node.args) > 1 or node.keywords):
                    refuse(file, node, "dynamic import options require a separate helper proof")
                module = literal(node.args[0]) if node.args else None
                if not isinstance(module, str) or module.startswith("."):
                    refuse(file, node, "unresolved dynamic import")
                else:
                    local_module(module, file, node)
            elif tail in ("exec", "eval", "compile", "spec_from_file_location", "SourceFileLoader",
                          "SourcelessFileLoader", "run_path", "run_module"):
                refuse(file, node, "dynamic code/file loading requires a separate input proof")
            elif (name.startswith("subprocess.") or name in ("os.system", "os.popen")
                  or name.startswith(("os.exec", "os.spawn", "multiprocessing.", "concurrent.futures."))):
                refuse(file, node, "subprocess helper dependencies are not statically closed")
            elif tail in ("read_text", "read_bytes", "open", "glob", "rglob", "iterdir", "listdir", "walk",
                          "iter_modules", "walk_packages", "FileIO", "file_digest", "load_module", "exec_module"):
                refuse(file, node, "dynamic file discovery/read requires a separate input proof")
            elif name in ("pickle.load", "pickle.loads", "cloudpickle.loads", "types.FunctionType", "marshal.loads"):
                refuse(file, node, "dynamic file discovery/read requires a separate input proof")
            elif tail == "getattr" and node.args and qualified(node.args[0]) in ("importlib", "builtins"):
                attribute = literal(node.args[1]) if len(node.args) > 1 else None
                if attribute in (None, "__import__", "import_module"):
                    refuse(file, node, "reflective import lookup")
    result["closure"] = dict(sorted(result["closure"].items()))
    result["status"] = "blocked" if result["unresolved"] else "proved"
    return result
