#!/usr/bin/env python3
"""Check explicit native execution closures, including registered ABI targets.

No provider/bindings exemption applies BELOW a native root. The public provider
may admit objects, pin owners, release/reacquire the GIL and publish results;
it may import a native root, never the reverse. std/max/layout/linalg are external compiler
packages (not recursively inspected); explicit std.python is still forbidden.
This is source enforcement, not runtime or identity verification. A declared
indirect ABI edge is followed just like an import, preventing a pure-looking
function-pointer facade from concealing a Python-dependent target.
"""
import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tokenize

MAMBA_ROOTS = (
    "mamba/impl/mamba_decode_plan.mojo",
    "mamba/native_block_dispatch.mojo",
    "mamba/native_state_dispatch.mojo",
    "mamba/native_decode_dispatch.mojo",
    "mamba/native_tape_dispatch.mojo",
)
GENERATION_ROOTS = (
    "transformer/native_generation_engine.mojo",
    "core/generation_policy_impl.mojo",
    "transformer/causal_lm_public_impl.mojo",
)
# This is a REAL current callback edge, not a permitted exception. Until its
# implementation is extracted and the provider-only GIL lifecycle is outside
# the execution loop, --scope all must fail on this target. Keep the mapping
# in sync with AdmittedNativeExecution.endpoint, not merely with filenames.
INDIRECT_TARGETS = {
    "core/native_execution.mojo": ("providers/python/native_execution_pyapi.mojo",),
}
EXTERNAL_PACKAGES = ("std", "max", "layout", "linalg")
PYTHON_NAMES = re.compile(r"(?:Python\w*|PyObject\w*|Py[A-Z]\w*|_Py\w*|GILReleased|GILAcquired)\Z")
CPYTHON_STRING = re.compile(r"[rubfRUBF]*([\"'])(?:_?Py[A-Z]\w*|_Py\w*)\1\Z")


def tokens(source):
    return list(tokenize.generate_tokens(io.StringIO(source).readline))


def imports(ts):
    result = []
    i = 0
    while i < len(ts):
        t = ts[i]
        if t.type != tokenize.NAME or t.string not in ("from", "import"):
            i += 1
            continue
        line = t.start[0]
        if t.string == "from":
            i += 1
            module = []
            while i < len(ts) and ts[i].string != "import" and ts[i].type != tokenize.NEWLINE:
                if ts[i].type in (tokenize.NAME, tokenize.OP):
                    module.append(ts[i].string)
                i += 1
            if i == len(ts) or ts[i].string != "import":
                continue
            i += 1
            names = []
            while i < len(ts) and ts[i].type != tokenize.NEWLINE:
                if ts[i].type == tokenize.NAME and ts[i].string != "as":
                    names.append(ts[i].string)
                i += 1
            result.append(("".join(module), names, line))
        else:
            i += 1
            module = []
            alias = False
            while i < len(ts) and ts[i].type != tokenize.NEWLINE:
                word = ts[i].string
                if word == ",":
                    result.append(("".join(module), [], line))
                    module, alias = [], False
                elif word == "as":
                    alias = True
                elif not alias and (ts[i].type == tokenize.NAME or word == "."):
                    module.append(word)
                i += 1
            result.append(("".join(module), [], line))
    return result


def resolve(module, names, importer, paths):
    relative = len(module) - len(module.lstrip("."))
    if relative:
        parent = importer.split("/")[:-1]
        if relative > len(parent) + 1:
            return []
        module = ".".join(parent[:len(parent) - relative + 1] + [module[relative:]])
    module = module.strip(".")
    base = module.replace(".", "/")
    found = []
    for candidate in (base + ".mojo", base + "/__init__.mojo"):
        if candidate in paths:
            found.append(candidate)
    for name in names:
        candidate = base + "/" + name + ".mojo"
        if candidate in paths:
            found.append(candidate)
        candidate = base + "/" + name + "/__init__.mojo"
        if candidate in paths:
            found.append(candidate)
    # Package initializers are dependencies too, even when the leaf resolves.
    components = base.split("/")
    for index in range(1, len(components)):
        candidate = "/".join(components[:index]) + "/__init__.mojo"
        if candidate in paths:
            found.append(candidate)
    return list(dict.fromkeys(found))


def inspect_closure(paths, read, roots, indirect):
    pending = [(root, [root]) for root in roots]
    visited, failures, edges, external = set(), [], [], set()
    while pending:
        path, chain = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        if path not in paths:
            failures.append(dict(path=path, chain=chain, reason="missing native source"))
            continue
        if path.startswith(("bindings/", "providers/")) or "pyapi" in Path(path).name or path.endswith("_py.mojo"):
            failures.append(dict(path=path, chain=chain, reason="provider/binding boundary reached from native execution"))
        try:
            ts = tokens(read(path))
        except (OSError, tokenize.TokenError, IndentationError) as problem:
            failures.append(dict(path=path, chain=chain, reason="cannot lex source: " + str(problem)))
            continue
        for token in ts:
            if ((token.type == tokenize.NAME and PYTHON_NAMES.fullmatch(token.string))
                    or (token.type == tokenize.STRING and CPYTHON_STRING.fullmatch(token.string))):
                failures.append(dict(path=path, line=token.start[0], chain=chain,
                                     reason="Python/CPython execution symbol: " + token.string))
        for module, names, line in imports(ts):
            if module == "python" or module.startswith(("python.", "std.python")):
                failures.append(dict(path=path, line=line, chain=chain, reason="Python import: " + module))
                continue
            if module.split(".")[0] in EXTERNAL_PACKAGES:
                external.add(module)
                continue
            children = resolve(module, names, path, paths)
            if not children:
                failures.append(dict(path=path, line=line, chain=chain, reason="unresolved repository import: " + module))
            for child in children:
                edges.append([path, child, "import"])
                pending.append((child, chain + [child]))
        for child in indirect.get(path, ()):
            edges.append([path, child, "declared indirect ABI target"])
            pending.append((child, chain + [child]))
    return dict(roots=list(roots), files=sorted(visited), edges=edges,
                external_compiler_imports=sorted(external), failures=failures,
                scope="repository source + declared ABI targets; external std/max/layout/linalg internals not inspected")


def self_test():
    safe = {"root.mojo": "from helper import f\n", "helper.mojo": "def f():\n    pass\n"}
    assert not inspect_closure(safe, safe.__getitem__, ["root.mojo"], {})["failures"]
    leaked = dict(safe, **{"helper.mojo": "from std.python import PythonObject\n"})
    assert any(len(x["chain"]) == 2 for x in inspect_closure(leaked, leaked.__getitem__, ["root.mojo"], {})["failures"])
    hidden = dict(safe, **{"providers/hidden.mojo": "def invoke():\n    pass\n"})
    assert inspect_closure(hidden, hidden.__getitem__, ["root.mojo"], {"helper.mojo": ["providers/hidden.mojo"]})["failures"]
    capi = {"root.mojo": 'def f():\n    var symbol = "PyObject_Call"\n'}
    assert inspect_closure(capi, capi.__getitem__, ["root.mojo"], {})["failures"]
    comment = {"root.mojo": '# from std.python import PythonObject\n"""PythonObject is forbidden below this boundary."""\n'}
    assert not inspect_closure(comment, comment.__getitem__, ["root.mojo"], {})["failures"]
    relative = {"pkg/root.mojo": "from . import helper\n", "pkg/helper.mojo": "from std.python import Python\n"}
    assert inspect_closure(relative, relative.__getitem__, ["pkg/root.mojo"], {})["failures"]
    print("native_neural_dependencies: six static positive/negative controls passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", default="HEAD")
    parser.add_argument("--worktree", action="store_true")
    parser.add_argument("--scope", choices=("mamba", "all"), default="all")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.worktree:
        listing = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"])
        paths = {x for x in listing.decode().split("\0") if x.endswith(".mojo")}
        # Sparse checkouts omit unchanged indexed files. Inspect their exact
        # indexed source, while retaining edited/untracked worktree bytes.
        def read(path):
            if Path(path).exists():
                return Path(path).read_text()
            return subprocess.check_output(["git", "show", ":" + path]).decode()
    else:
        listing = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "-z", args.tree])
        paths = {x for x in listing.decode().split("\0") if x.endswith(".mojo")}
        read = lambda path: subprocess.check_output(["git", "show", args.tree + ":" + path]).decode()
    roots = tuple(args.root) if args.root else MAMBA_ROOTS + (() if args.scope == "mamba" else GENERATION_ROOTS)
    report = inspect_closure(paths, read, roots, INDIRECT_TARGETS)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    print(f"native_neural_dependencies: {len(report['files'])} source files, {len(report['edges'])} edges, {len(report['failures'])} findings")
    for failure in report["failures"][:30]:
        print(f"{failure['path']}:{failure.get('line', 0)}: {failure['reason']}; chain: {' -> '.join(failure['chain'])}")
    if len(report["failures"]) > 30:
        print("Additional findings retained in --json report; none treated as passing.")
    print(report["scope"])
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
