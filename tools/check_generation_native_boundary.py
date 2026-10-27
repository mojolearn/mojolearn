#!/usr/bin/env python3
"""Check the new generation native modules' local source dependency closure.

This is an architectural regression check, not ABI/runtime qualification. It
does not inspect the compiled implementation of upstream standard libraries.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ("transformer.session_abi", "transformer.native_session_call",
         "transformer.session_native_dispatch", "core.native_descriptor_abi",
         "core.native_descriptor_call", "training.native_primitive_dispatch",
         "mamba.native_state_dispatch")
FORBIDDEN_IMPORTS = ("std.python", "providers", "bindings")
CALLBACK_TOKENS = re.compile(r"\b(?:PythonObject|PyObjectPtr|Python|py_call|py_call_typed|GILReleased)\b")
# Resolved from Modular's installed native packages, not this repository.
EXTERNAL_MODULES = {"layout", "layout.tile_layout", "linalg.gemv", "linalg.matmul"}


def code_only(source):
    source = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', "", source, flags=re.S)
    return re.sub(r"#.*", "", source)


def local_file(root, module):
    path = root.joinpath(*module.split("."))
    for candidate in (path.with_suffix(".mojo"), path / "__init__.mojo"):
        if candidate.is_file():
            return candidate
    return None


def audit(root=ROOT, roots=ROOTS):
    queue = [(module, (module,)) for module in roots]
    seen, violations = set(), []
    while queue:
        module, chain = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        path = local_file(root, module)
        if path is None:
            if not module.startswith(("std.", "max.")) and module not in EXTERNAL_MODULES:
                violations.append((chain, "unresolved local dependency"))
            continue
        code = code_only(path.read_text())
        without_strings = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', "", code)
        if CALLBACK_TOKENS.search(without_strings) or re.search(r'["\']Py(?:Object|CFunction|Capsule|Import|Err|_)[A-Za-z_]*["\']', code):
            violations.append((chain, "Python/CPython helper token in native code"))
        imports = re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", code, re.M)
        for imported in imports:
            if any(imported == prefix or imported.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORTS):
                violations.append((chain + (imported,), "provider/Python dependency"))
            else:
                queue.append((imported, chain + (imported,)))
    return seen, violations


def main():
    seen, violations = audit()
    for chain, reason in violations:
        print(f"FAIL {reason}: {' -> '.join(chain)}")
    print(f"generation native closure: {len(seen)} modules, {len(violations)} violations")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
