#!/usr/bin/env python3
"""Source guard for explicit Python interpreter execution in product Mojo.

Andrew's 2026-10-28 rule permits native CPython access anywhere, including
loops. Imports, PythonObject values, refcount/GIL APIs and PyObject_Call are
not violations by spelling. In particular PyObject_Call may target an admitted
native function; its target and possible hooks require execution-path review.

This deliberately narrow lexical guard rejects recognized calls that explicitly
evaluate Python source/code (Python.evaluate and PyRun/PyEval entry points).
It scans bindings and providers too: moving an algorithm into those directories
cannot exempt it. Tooling/fixtures under bench, experiments, tests and checks
are outside this product-source scan. This is NOT a transitive callback-safety
proof: aliases, dynamic targets, user hooks and Python-library computation need
admission review and the separate runtime-endpoint ledger/source audit.

The command/status name is retained for hook/CI compatibility. Modes:
  --tree [REF] [--branch] [--baseline FILE]
      Judge committed source, using a shrinking explicit-execution exception
      baseline. --branch excludes unchanged pre-existing findings.
  --prune-baseline [REF]
      Remove stale explicit-execution baseline rows, never add them.
The old python_in_kernels_baseline.txt remains historical import-debt evidence;
it is not authorization to execute Python. Exit 0/1/2: clean/findings/usage.
No product import, Mojo build or runtime execution is performed.
"""
import ast
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = "tools/hooks/python_execution_baseline.txt"
TOOLING_PREFIXES = ("bench/", "experiments/", "tests/", "tools/", "checks/")
# These APIs execute source/code by contract. PyObject_Call and generic object
# protocols are intentionally absent: exact admitted targets can be native.
EVALUATORS = frozenset({
    "PyRun_SimpleString", "PyRun_SimpleStringFlags", "PyRun_String", "PyRun_StringFlags",
    "PyRun_SimpleFile", "PyRun_SimpleFileEx", "PyRun_SimpleFileExFlags",
    "PyRun_File", "PyRun_FileEx", "PyRun_FileFlags", "PyRun_FileExFlags",
    "PyRun_AnyFile", "PyRun_AnyFileEx", "PyRun_AnyFileFlags", "PyRun_AnyFileExFlags",
    "PyRun_InteractiveOne", "PyRun_InteractiveOneFlags", "PyRun_InteractiveOneObject",
    "PyRun_InteractiveLoop", "PyRun_InteractiveLoopFlags",
    "PyEval_EvalCode", "PyEval_EvalCodeEx", "PyEval_EvalFrame", "PyEval_EvalFrameEx",
    "_PyEval_EvalFrameDefault",
})


def is_tooling(path):
    return path.startswith(TOOLING_PREFIXES) or "/checks/" in path



# A small lexical scanner, not a Mojo parser. Keep strings as single tokens so
# a C symbol literal can be recognized at a call site; discard comments and
# never scan the contents of documentation or arbitrary strings as source.
_TOKEN = re.compile(
    r'(?P<comment>\#[^\n]*)|'
    r'(?P<string>(?:[rRuUbBfF]{0,2})(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|'
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'))|'
    r'(?P<name>[A-Za-z_]\w*)|(?P<newline>\n)|(?P<other>[^\s])'
)


def _tokens(source):
    tokens, line, position = [], 1, 0
    for match in _TOKEN.finditer(source):
        line += source.count('\n', position, match.start())
        if match.lastgroup != 'comment':
            tokens.append((match.group(), line, match.lastgroup))
        line += source.count('\n', match.start(), match.end())
        position = match.end()
    return tokens


def _after_group(tokens, start, opening, closing):
    if start >= len(tokens) or tokens[start][0] != opening:
        return None
    depth = 0
    for i in range(start, len(tokens)):
        if tokens[i][0] == opening:
            depth += 1
        elif tokens[i][0] == closing:
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _literal(token):
    if token[2] != 'string':
        return None
    try:
        value = ast.literal_eval(token[0])
        return value if isinstance(value, str) else None
    except (ValueError, SyntaxError):
        return None


def _imports(tokens):
    python_classes, cpython_modules, functions = set(), set(), {}
    externals = {('std', 'ffi', 'external_call')}
    for i, token in enumerate(tokens):
        if token[0] not in ('from', 'import'):
            continue
        j = i + 1
        parts = []
        while j < len(tokens) and (tokens[j][2] == 'name' or tokens[j][0] == '.'):
            if tokens[j][0] in ('import', 'as'):
                break
            if tokens[j][0] != '.':
                parts.append(tokens[j][0])
            j += 1
        module = '.'.join(parts)
        if token[0] == 'import':
            alias = tuple(parts)
            if j + 1 < len(tokens) and tokens[j][0] == 'as':
                alias = (tokens[j + 1][0],)
            if module in ('std.python', 'python'):
                python_classes.add(alias + ('Python',))
            elif module in ('std.python._cpython', 'python._cpython'):
                cpython_modules.add(alias)
            elif module == 'std.ffi':
                externals.add(alias + ('external_call',))
            continue
        if j >= len(tokens) or tokens[j][0] != 'import':
            continue
        j += 1
        grouped = j < len(tokens) and tokens[j][0] == '('
        if grouped:
            j += 1
        while j < len(tokens):
            if tokens[j][0] == ')' or (not grouped and tokens[j][0] == '\n'):
                break
            if tokens[j][2] != 'name':
                j += 1
                continue
            name = alias = tokens[j][0]
            j += 1
            if j + 1 < len(tokens) and tokens[j][0] == 'as':
                alias = tokens[j + 1][0]
                j += 2
            if module in ('std.python', 'python') and name == 'Python':
                python_classes.add((alias,))
            if module in ('std.python', 'python') and name == '_cpython':
                cpython_modules.add((alias,))
            if module in ('std.python._cpython', 'python._cpython') and name in EVALUATORS:
                functions[(alias,)] = name
            if module == 'std.ffi' and name == 'external_call':
                externals.add((alias,))
    return python_classes, cpython_modules, functions, externals


def execution_findings(source):
    """Recognized evaluator invocations as (line, target), not call safety.

    Simple import aliases are recognized. This does not resolve assignment
    aliases, shadowing, function pointers or dynamic callable provenance.
    """
    if not re.search(r"evaluate|PyRun_|PyEval_|_PyEval_", source):
        return []
    original = _tokens(source)
    python_classes, cpython_modules, functions, externals = _imports(original)
    tokens = [token for token in original if token[2] != 'newline']
    findings = []
    for i, token in enumerate(tokens):
        if token[2] != 'name' or (i > 0 and tokens[i - 1][0] == '.'):
            continue
        chain = [token[0]]
        j = i + 1
        while j + 1 < len(tokens) and tokens[j][0] == '.' and tokens[j + 1][2] == 'name':
            chain.append(tokens[j + 1][0])
            j += 2
        chain = tuple(chain)
        callee = None
        if j < len(tokens) and tokens[j][0] == '(':
            if chain[:-1] in python_classes and chain[-1] == 'evaluate':
                callee = '.'.join(chain)
            elif chain in functions:
                callee = functions[chain]
            elif chain[:-1] in cpython_modules and chain[-1] in EVALUATORS:
                callee = chain[-1]
        if chain in externals:
            end = _after_group(tokens, j, '[', ']')
            if end is not None and end < len(tokens) and tokens[end][0] == '(' and j + 1 < len(tokens):
                name = _literal(tokens[j + 1])
                if name in EVALUATORS:
                    callee = name
        if chain[-2:] == ('lib', 'get_function') and chain[:-2] in cpython_modules:
            end = _after_group(tokens, j, '[', ']')
            arguments = _after_group(tokens, end, '(', ')') if end is not None else None
            if arguments is not None and arguments < len(tokens) and tokens[arguments][0] == '(':
                name = _literal(tokens[end + 1])
                if name in EVALUATORS:
                    callee = name
        if callee:
            findings.append((token[1], callee))
    return findings


def _git(*args, inp=None):
    return subprocess.run(["git", *args], capture_output=True, input=inp)


def _resolve(ref):
    r = _git("rev-parse", "--verify", "--quiet", ref + "^{commit}")
    return r.stdout.decode().strip() if r.returncode == 0 else None


def _mojo_blobs(ref):
    """{path: blob sha} of product Mojo, including every provider/binding."""
    r = _git("ls-tree", "-r", "--full-tree", ref)
    if r.returncode != 0:
        raise SystemExit(f"no_python_in_kernels: cannot list {ref}: {r.stderr.decode().strip()}")
    out = {}
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        meta, _, path = ln.partition("\t")
        bits = meta.split()
        if len(bits) == 3 and bits[1] == "blob" and path.endswith(".mojo") and not is_tooling(path):
            out[path] = bits[2]
    return out


def _read(shas):
    """{sha: text}, one `git cat-file --batch` process."""
    need = list(dict.fromkeys(shas))
    if not need:
        return {}
    out = _git("cat-file", "--batch", inp="".join(s + "\n" for s in need).encode()).stdout
    texts, i = {}, 0
    for sha in need:
        nl = out.index(b"\n", i)
        head = out[i:nl].split()
        i = nl + 1
        if len(head) < 3 or head[1] == b"missing":
            texts[sha] = ""
            continue
        size = int(head[2])
        texts[sha] = out[i:i + size].decode("utf-8", "replace")
        i += size + 1
    return texts


def findings_at(ref, paths=None):
    """Paths with recognized explicit evaluator calls (not an import/purity count)."""
    blobs = _mojo_blobs(ref)
    if paths is not None:
        blobs = {p: s for p, s in blobs.items() if p in paths}
    texts = _read(blobs.values())
    return {p: found for p, s in blobs.items() if (found := execution_findings(texts[s]))}


def hits_at(ref, paths=None):
    return sorted(findings_at(ref, paths))


def parse_baseline(text):
    return {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")}


def _baseline_at(ref):
    r = _git("show", f"{ref}:{BASELINE}")
    return parse_baseline(r.stdout.decode("utf-8", "replace")) if r.returncode == 0 else None


def _main_ref():
    for m in ("refs/remotes/origin/main", "origin/main", "refs/heads/main"):
        if _resolve(m):
            return m
    return None


def _per_dir(paths):
    c = Counter(p.split("/", 1)[0] if "/" in p else "." for p in paths)
    return ", ".join(f"{d} {n}" for d, n in sorted(c.items())) or "none"


def check_tree(ref, branch=False, baseline_file=None):
    sha = _resolve(ref)
    if not sha:
        print(f"no_python_in_kernels: unknown ref {ref}", file=sys.stderr)
        return 2
    if baseline_file:
        with open(baseline_file, encoding="utf-8") as f:
            base = parse_baseline(f.read())
    else:
        base = _baseline_at(sha) or set()
    main = _main_ref()
    main_base = (_baseline_at(main) or set()) if main else None
    if main_base is not None:
        grown = sorted(base - main_base)
        base &= main_base  # the baseline only shrinks: rows main lacks are not grandfathered
    else:
        grown = []
    details = findings_at(sha)
    hits = sorted(details)
    judged, old = set(hits), {}
    if branch and main:
        mb = _git("merge-base", main, sha).stdout.decode().strip()
        if mb and mb != sha:
            changed = _git("diff", "--name-only", "--no-renames", mb, sha, "--", "*.mojo").stdout.decode().split()
            judged &= set(changed)
            old = findings_at(mb, judged)
    additions = {}
    for path in judged - base:
        prior = Counter(callee for _, callee in old.get(path, ()))
        for finding in details[path]:
            if prior[finding[1]]:
                prior[finding[1]] -= 1
            else:
                additions.setdefault(path, []).append(finding)
    new = sorted(additions)
    stale = sorted(base - set(hits)) if not branch else []
    print(f"no_python_in_kernels: {len(hits)} product .mojo files contain explicit Python evaluator calls at {sha[:9]} "
          f"(baseline {len(base)}); per directory: {_per_dir(hits)}", file=sys.stderr)
    for p in grown:
        print(f"no_python_in_kernels: baseline row {p} is not in main's baseline; the baseline only shrinks",
              file=sys.stderr)
    for p in stale:
        print(f"no_python_in_kernels: stale baseline row {p} (no longer has a recognized evaluator call): "
              f"remove it with --prune-baseline", file=sys.stderr)
    for p in new:
        for line, callee in additions[p]:
            print(f"{p}:{line}: explicit Python evaluator call {callee}; native C-API access is allowed, "
                  "Python source/code execution requires a documented exception", file=sys.stderr)
    print("no_python_in_kernels: narrow source check only; dynamic calls/hooks require target and "
          "type-admission review (runtime-endpoint ledger remains separate)", file=sys.stderr)
    return 1 if new or grown else 0


def prune_baseline(ref):
    sha = _resolve(ref)
    if not sha:
        print(f"no_python_in_kernels: unknown ref {ref}", file=sys.stderr)
        return 2
    top = _git("rev-parse", "--show-toplevel").stdout.decode().strip()
    path = os.path.join(top, BASELINE)
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    hits = set(hits_at(sha))
    keep = [ln for ln in lines if not ln.strip() or ln.lstrip().startswith("#") or ln.strip() in hits]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(keep) + "\n")
    print(f"no_python_in_kernels: removed {len(lines) - len(keep)} stale rows from {BASELINE}", file=sys.stderr)
    return 0


def main(argv):
    args = list(argv)
    branch = "--branch" in args
    args = [a for a in args if a != "--branch"]
    baseline_file = None
    if "--baseline" in args:
        i = args.index("--baseline")
        if i + 1 >= len(args):
            print(__doc__, file=sys.stderr)
            return 2
        baseline_file = args[i + 1]
        del args[i:i + 2]
    if args and args[0] == "--tree" and len(args) <= 2:
        return check_tree(args[1] if len(args) == 2 else "HEAD", branch, baseline_file)
    if args and args[0] == "--prune-baseline" and len(args) <= 2:
        return prune_baseline(args[1] if len(args) == 2 else "HEAD")
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
