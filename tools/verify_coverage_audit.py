#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Which new or changed endpoints does the verifier cover, and which have a sabotage arm (release gate, 2026-10-21).

The gate: every native entry and every public Python method that is new or
changed since the last published release must be reached by a verifier lane
(tools/identity_break.py, tools/identity_lanes/*.py) and by a fault-injection
arm the verifier or a check provably catches. This tool builds that list from
the code, derived the way `tools/lane_select.py` derives lane closures, and
never from a hand-kept list:

  native entries  every `def_function` / `def_py_function` / `def_py_c_function`
                  registration under bindings/ at HEAD. NEW when the module did
                  not register the name at BASE; CHANGED when a Mojo definition
                  in the export's closure (`lane_select._MojoDefGraph`, the same
                  graph `mojo_edit_lanes` selects with) differs from BASE. A
                  definition whose text equals a same-named definition at BASE
                  in any file is a MOVE and does not count. The kind of the
                  first changed definition is `door` (bindings/, *pyapi*, *_py),
                  `impl` (*_impl*) or `kernel`.
  lanes           `lane_select.binding_users(<binding>, exports=[name])`: a lane
                  whose recorded binding use calls the export (`precise`), or
                  that is declared for / runs the binding whole (`wide`).
  sabotage        a definition in the export's closure that reads a
                  `MOJOLEARN_*SABOTAGE*` define (HOST_SABOTAGE is the CPU
                  identity gate's arm, which reruns the verifier lanes and
                  requires their parts to move), or one that a
                  `*/sabotage/*.patch` edits (the per-seam check arms).
  Python methods  every public operation method of `tools/public_runtime_inventory.py`
                  whose walked call path (the method and the mojolearn helpers it
                  reaches) has lines changed since BASE. Lanes: the lanes covering
                  the native entries the walk reaches, among the lanes whose
                  symbol closure names the public class.

    python3 tools/verify_coverage_audit.py --base fc8d425fb --json ~/mojolearn-evidence/verify-coverage-20261021/audit.json
    python3 tools/verify_coverage_audit.py --from-json <audit.json> --markdown docs/plans/VERIFY_COVERAGE_20261021.md

Out of scope by design: the Torch and MAX adapters (adapters/, python/mojolearn/adapters/, the
wheels' _adapters/) are the SAME API in another format: front-ends over the kernels the Python API
calls, computing nothing of their own. They get NO identity lanes, NO verifier rows and NO sabotage
arms (Andrew, 2026-10-24; AGENTS.md "Framework adapters are the same API in another format"). Their
qualification is the installed-wheel smoke on each vendor (tools/adapter_wheel_smoke.py, run by the
release gate).
Static analysis only: nothing is imported from mojolearn, nothing is built or run.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from mojo_registrations import registered_functions
from verify_reference_evidence import reference_label, reference_snapshot, unknown_references
SAB_RE = re.compile(r"MOJOLEARN_[A-Z0-9_]*SABOTAGE[A-Z0-9_]*")
#: The framework adapters' Python package. An entry whose only Python callers
#: live here is the adapters' own: no identity lane, verifier row or sabotage
#: arm (Andrew, 2026-10-24; the module docstring's "Out of scope by design").
ADAPTER_PACKAGE = "python/mojolearn/adapters/"
ADAPTER_ONLY = ("adapter only: its only Python callers are the framework adapters (python/mojolearn/adapters/), "
                "the same API in another format, qualified by the installed-wheel adapter smoke, never by a lane "
                "(AGENTS.md, Andrew 2026-10-24)")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=False).stdout


def show(rev, path):
    out = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{path}"], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def registrations(text):
    """name -> (impl, [param identifiers]) for one binding source."""
    return registered_functions(text)


def kind_of(path):
    base = os.path.basename(path)
    if path.startswith("bindings/") or "pyapi" in base or base.endswith("_py.mojo"):
        return "door"
    if "_impl" in base:
        return "impl"
    return "kernel"


def changed_mojo_nodes(ls, base):
    """{node} of HEAD definitions that differ from BASE, moves excluded; plus
    the set of files read whole (their every unit counts as changed)."""
    files = [p for p in git("diff", "--no-renames", "--name-only", f"{base}..HEAD", "--", "*.mojo").split() if p]
    status = {}
    for line in git("diff", "--no-renames", "--name-status", f"{base}..HEAD", "--", "*.mojo").splitlines():
        parts = line.split("\t")
        status[parts[-1]] = parts[0]
    old_texts = defaultdict(set)          # unit key without file -> texts at BASE, over EVERY BASE Mojo file
    for old in base_mojo_texts(base):
        units = ls._mojo_units(old)
        if units is None:
            continue
        for key, text in ls._mojo_unit_texts(units).items():
            old_texts[key].add(text.strip())
    nodes, whole = set(), set()
    for path in files:
        if not os.path.exists(os.path.join(ROOT, path)):
            continue
        new = ls._read(path)
        old = show(base, path)
        changed, _ = ls.mojo_changed_definitions(old, new)
        units = ls._mojo_units(new)
        if units is None:
            whole.add(path)
            nodes.add(("f", path))
            continue
        texts = ls._mojo_unit_texts(units)
        if changed is None:
            changed = set(texts)
        for unit in changed:
            if unit not in texts:
                continue                  # removed: its callers are marked rebound by lane_select
            if texts[unit].strip() in old_texts.get(unit, ()):
                continue                  # a move: the same text existed at BASE
            nodes.add((unit[0], path) + unit[1:])
    return nodes, whole, status


def base_mojo_texts(base):
    """The text of every Mojo file at BASE (one `git cat-file --batch`)."""
    paths = [p for p in git("ls-tree", "-r", "--name-only", base).splitlines() if p.endswith(".mojo")]
    req = "".join(f"{base}:{p}\n" for p in paths)
    out = subprocess.run(["git", "-C", ROOT, "cat-file", "--batch"], input=req.encode(), capture_output=True).stdout
    texts, i = [], 0
    while i < len(out):
        nl = out.index(b"\n", i)
        head = out[i:nl].split()
        if len(head) < 3 or head[1] != b"blob":
            i = nl + 1
            continue
        size = int(head[2])
        texts.append(out[nl + 1:nl + 1 + size].decode("utf-8", "replace"))
        i = nl + 1 + size + 1
    return texts


def patch_nodes(ls):
    """node -> [patch path] for every */sabotage/*.patch: the HEAD definitions
    holding a line the patch removes."""
    out = defaultdict(list)
    found = git("ls-files", "*/sabotage/*.patch", "*/sabotage/**/*.patch").split()
    for patch in sorted(set(found)):
        try:
            text = open(os.path.join(ROOT, patch)).read()
        except OSError:
            continue
        # per hunk, the lines it removes, or, for a hunk that only adds, its
        # context lines: the definition holding them is the one patched
        target, removed, hunk = None, defaultdict(list), None

        def close(h):
            if h and target:
                removed[target].extend(h["minus"] or h["ctx"])

        for line in text.splitlines():
            if line.startswith("--- "):
                close(hunk)
                hunk = None
            elif line.startswith("+++ "):
                target = line[4:].strip()
                target = target[2:] if target.startswith("b/") else target
            elif line.startswith("@@"):
                close(hunk)
                hunk = dict(minus=[], ctx=[])
            elif hunk is not None and line.startswith("-"):
                if len(line[1:].strip()) > 6:
                    hunk["minus"].append(line[1:].strip())
            elif hunk is not None and line.startswith(" "):
                if len(line[1:].strip()) > 6:
                    hunk["ctx"].append(line[1:].strip())
        close(hunk)
        for tgt, lines in removed.items():
            if not tgt.endswith(".mojo") or not os.path.exists(os.path.join(ROOT, tgt)):
                continue
            units = ls._mojo_units(ls._read(tgt))
            if units is None:
                out[("f", tgt)].append(patch)
                continue
            for unit, utext in ls._mojo_unit_texts(units).items():
                stripped = {ln.strip() for ln in utext.splitlines()}
                if any(ln in stripped for ln in lines):
                    out[(unit[0], tgt) + unit[1:]].append(patch)
    return out


def label(node):
    return f"{node[1]} (whole file)" if node[0] == "f" else (
        f"{node[1]}:{node[2]}" if node[0] == "t" else f"{node[1]}:{node[2]}.{node[3]}")


def native_rows(ls, base, reach, whole_surface=False):
    changed, whole, status = changed_mojo_nodes(ls, base)
    pnodes = patch_nodes(ls)
    graph = ls._MojoDefGraph()
    srcs = sorted(p for p in git("ls-files", "bindings/*.mojo", "bindings/**/*.mojo").split())
    py_words = defaultdict(int)
    # The framework adapters' Python callers are counted apart: an entry only
    # they call is the adapters' (ADAPTER_ONLY), never a GAP (AGENTS.md
    # "Framework adapters are the same API in another format").
    adapter_words = defaultdict(int)
    for p in git("ls-files", "python/mojolearn").split():
        base_name = os.path.basename(p)
        if p.endswith(".py") and "/tests/" not in p and base_name != "host_surface.py" \
                and not base_name.startswith("_surface_"):
            # code only, never a comment or a docstring: an attribute, a name,
            # or a string that is not a docstring
            try:
                tree = ast.parse(open(os.path.join(ROOT, p)).read())
            except (OSError, SyntaxError):
                continue
            docs = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                        and node.body and isinstance(node.body[0], ast.Expr) \
                        and isinstance(node.body[0].value, ast.Constant):
                    docs.add(id(node.body[0].value))
            words = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    words.add(node.attr)
                elif isinstance(node, ast.Name):
                    words.add(node.id)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                    words |= set(IDENT_RE.findall(node.value)) if len(node.value) < 80 else set()
            counts = adapter_words if p.startswith(ADAPTER_PACKAGE) else py_words
            for w in words:
                counts[w] += 1
    anywhere = set()
    for p in git("ls-tree", "-r", "--name-only", base, "bindings/").split():
        if p.endswith(".mojo"):
            anywhere |= set(registrations(show(base, p)))
    rows = []
    for src in srcs:
        regs = registrations(ls._read(src))
        if not regs:
            continue
        old = registrations(show(base, src)) if status.get(src) != "A" else {}
        name = os.path.basename(src)[:-5]
        for export, (impl, params) in sorted(regs.items()):
            roots = set(graph.resolve(src, impl))
            for p in params:
                roots |= graph.resolve(src, p)
            parent = closure(graph, roots, ls)
            reached = set(parent)
            hits = sorted(reached & changed, key=label)
            new = export not in old
            if not new and not hits and not whole_surface:
                continue
            kinds = sorted({kind_of(n[1]) for n in hits})
            sab_defs, sab_sites, patches = set(), [], set()
            for n in reached:
                text = graph.text(n)
                defs = set(SAB_RE.findall(text)) - {"MOJOLEARN_HOST_ALLOW_SABOTAGE"}
                if defs:
                    sab_defs |= defs
                    sab_sites.append(label(n))
                for p in pnodes.get(n, ()):
                    patches.add(p)
            users = ls.binding_users(name, exports=[export])
            precise = {lane for lane, (_, ents, _) in reach.items() if NameClosure.names_entry(ents, export)}
            # an owner entry whose own impl file holds a dispatch branch the
            # lane reaches (OwnerDispatch) is reached by the lane
            files = {n[1] for n in roots}
            precise |= {lane for lane, owned in OWNER_FILES.items() if owned & files}
            probe = any(constant_probe(graph.text(n)) for n in graph.resolve(src, impl))
            rows.append(dict(entry=export, src=src, impl=impl, new=new, new_anywhere=export not in anywhere, py_files=py_words.get(export, 0), adapter_files=adapter_words.get(export, 0), unresolved=not roots, constant_probe=probe,
                             kinds=kinds, changed=[label(n) for n in hits[:4]], n_changed=len(hits),
                             changed_kernel=[label(n) for n in hits if kind_of(n[1]) != "door"],
                             # in the closure's breadth-first order: the kernels nearest the export first
                             closure_kernel=[label(n) for n in parent if kind_of(n[1]) != "door"
                                             and _sab_na(label(n)) is None
                                             and not _valueless(graph.text(n), n)][:4],
                             n_closure_kernel=sum(1 for n in reached if kind_of(n[1]) != "door"
                                                  and not _valueless(graph.text(n), n)),
                             lanes_precise=sorted(precise), lanes_wide=sorted(set(users) - precise),
                             lanes_closure=len(users),
                             sabotage_defines=sorted(sab_defs), sabotage_sites=sorted(sab_sites)[:4],
                             sabotage_patches=sorted(patches)))
    return rows


_PROBE_RETURN = re.compile(r'^return PythonObject\((?:-?\d+(?:\.\d+)?|True|False|None|"[^"]*"|'
                           r'String\((?:"[^"]*"|[A-Z][A-Z0-9_]*)\)|[A-Z][A-Z0-9_]*)\)$')


_VALUELESS_HEAD = re.compile(r"^\s*(?:@[A-Za-z_][\w.]*(?:\[[^\]]*\])?(?:\([^)]*\))?\s+)*"
                             r"(?:comptime|alias|struct|trait|var)\b")


#: A type's lifecycle and printing methods: construction, copies, moves,
#: teardown, comparison and repr hold no arithmetic a lane's digest carries.
_VALUELESS_METHODS = frozenset({"__init__", "__deinit__", "__moveinit__", "__copyinit__", "__del__",
                                "__eq__", "__ne__", "__hash__", "__bool__", "__len__", "write_to",
                                "write_repr_to", "copy"})


def _valueless(text, node=None):
    """A definition with no arithmetic to fault: a compile-time constant or
    alias (a define, a profile word, a slot name), a type (a handle, a
    registry, a session struct: its methods are separate nodes), or a type's
    lifecycle or printing method."""
    if node is not None and node[0] == "m" and node[3] in _VALUELESS_METHODS:
        return True
    return bool(text) and bool(_VALUELESS_HEAD.match(text))


def constant_probe(text):
    """True for a registration whose whole body returns one constant: a
    contract-version probe (`kmeans_parallel_available` -> 1), a compiled
    vendor or flag (`preprocessing_vendor` -> COMPILED_VENDOR). It reads no
    data and computes nothing, so no lane digest can carry it."""
    if not text:
        return False
    body = re.sub(r'"""[\s\S]*?"""', "", text)
    lines = [ln.split("#", 1)[0].strip() for ln in body.splitlines()[1:]]
    lines = [ln for ln in lines if ln]
    return len(lines) == 1 and bool(_PROBE_RETURN.match(lines[0]))


def closure(graph, roots, ls):
    """The closure `_MojoDefGraph.closure` computes, from explicit roots (the
    graph's own reads `def_function` only; this also takes `def_py_function`
    and `def_py_c_function`)."""
    parent, queue, mentioned, waiting = {}, [], set(), {}

    def visit(node, came):
        if node not in parent:
            parent[node] = came
            queue.append(node)

    for r in sorted(roots):
        visit(r, None)
    k = 0
    while k < len(queue):
        node = queue[k]
        k += 1
        if node[0] == "t":
            units = graph.units(node[1])
            for meth in sorted((units or {}).get("methods", {}).get(node[2], {})):
                m = ("m", node[1], node[2], meth)
                if meth in mentioned or meth in ls._MOJO_IMPLICIT_METHODS or (
                        meth.startswith("__") and meth.endswith("__")):
                    visit(m, node)
                else:
                    waiting.setdefault(meth, []).append((m, node))
        fresh = graph.idents(node) - mentioned
        mentioned |= fresh
        for ident in sorted(fresh):
            for m, owner in waiting.pop(ident, ()):
                visit(m, owner)
        for t in graph.targets(node):
            visit(t, node)
    return parent


def changed_python_lines(base):
    out = defaultdict(set)
    path = None
    for line in git("diff", "-U0", f"{base}..HEAD", "--", "python/mojolearn/*.py",
                    "python/mojolearn/**/*.py").splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip()
            path = p[2:] if p.startswith("b/") else None
        elif line.startswith("@@") and path:
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            start, count = int(m.group(1)), int(m.group(2) or 1)
            out[path] |= set(range(start, start + max(count, 1)))
    return {p: s for p, s in out.items() if "/tests/" not in p}


#: Operations the harness itself runs on every lane's fitted estimator (the
#: model column saves it and loads it back: identity_break._probe_fit_impl).
HARNESS_OPS = frozenset({"save", "load"})


class Walker:
    """The public_runtime_inventory walk, keeping the functions it reaches."""

    def __init__(self):
        import public_runtime_inventory as pri
        self.pri = pri
        self.index = pri.Index(pri.REPO, "mojolearn")
        self.cache, self.walks = {}, {}
        self.by_name = defaultdict(list)          # bare name -> [("func", Func) | ("class", Cls)]
        for mod in [m for m in self.index.mods.values() if m is not None]:
            for node in mod.tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.by_name[node.name].append(("func", self.index.func(mod, node)))
                elif isinstance(node, ast.ClassDef):
                    self.by_name[node.name].append(("class", self.index.cls(mod, node)))

    def walk(self, func, cls):
        key0 = (func.key, cls.key if cls else None)
        if key0 in self.walks:
            return self.walks[key0]
        seen, stack, entries, funcs = set(), [(func, cls)], set(), []
        while stack:
            item, klass = stack.pop()
            key = (item.key, klass.key if klass else None)
            if key in seen:
                continue
            seen.add(key)
            funcs.append(item)
            if key not in self.cache:
                try:
                    self.cache[key] = self.pri.analyze(self.index, item, klass)
                except Exception:          # one unreadable helper does not stop the audit
                    self.cache[key] = self.pri.Facts()
            facts = self.cache[key]
            entries |= facts.entries
            stack.extend(facts.callees)
        self.walks[key0] = (funcs, entries)
        return funcs, entries

    def ops(self, cls, names):
        """(op, Func) for the operation methods of `cls` a lane can call."""
        out = []
        # Enumerate the class, not the old estimator-only operation vocabulary.
        # Otherwise backward, checkpoint IO, properties and parameter methods
        # cannot be credited even when the lane explicitly calls them.
        members = {n.name for base in self.index.mro(cls) for n in base.node.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and self.pri._public_member(n.name)}
        for op in sorted(members):
            if op not in names and op not in HARNESS_OPS and op != "__init__":
                continue
            found = self.index.lookup(cls, op)
            if found and found[0] == "func":
                out.append((op, found[1]))
        return out


class ReachedSet(set):
    """A name closure's reached names, with the string constants (`.strings`)
    and class names (`.classes`) among them kept apart: a native owner's
    dispatch value is matched against those, never against every identifier."""
    strings = frozenset()
    classes = frozenset()


class NameClosure:
    """A scoped definition closure over the package, the second reach. The
    walker resolves calls by type and stops at what it cannot type (`k.ew(...)`
    on a local kit object, an entry fetched by a string); this one resolves a
    bare name or a module attribute the way Python does (the module's own
    definitions and imports, `public_runtime_inventory.Index`), and follows a
    method of a reached class when its name is reached as an attribute or a
    string anywhere in the closure (or it is a dunder, or the harness's
    save/load). Import-time calls of a reached module come with it. Every
    attribute, name and identifier-shaped string a followed definition holds
    is a reached name; binding entries are matched against those."""

    IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    SUFFIX = "\x00suffix"
    _suffixes = {}
    OPS_NAMES = frozenset({"fit", "predict", "transform", "score", "save", "load", "step", "forward",
                           "generate", "sample", "run", "close", "open", "main", "call"})

    def __init__(self, walker):
        self.index = walker.index
        self.by_name = walker.by_name
        self.methods = {}                    # Cls.key -> {name: Func}
        self.top = {}                        # module -> [Func] it calls at import
        self.bound_calls = {}                # module -> {name: [("func", Func)] its assignment calls}
        self.bound_strings = {}              # module -> {name: the string arguments of that call}
        for mod in [m for m in self.index.mods.values() if m is not None]:
            local = mod.defs
            calls = []
            for node in mod.tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                     ast.Import, ast.ImportFrom)):
                    continue
                if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
                    continue
                for n in ast.walk(node):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in local:
                        calls.append(self.index.func(mod, local[n.func.id]))
            self.top[mod.name] = calls
            # `_IPCA_CALL, _IPCA_CONTEXT = _native_ipca_bridge.connect_public(IncrementalPCA)`
            # (_expansion_decomp.py): a migrated method calls the module-level
            # name the bridge's connect returned, so reading the name follows
            # the connect (which names the owner entry, `ipca_native_call`)
            bound = {}
            for node in mod.tree.body:
                if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                    continue
                try:
                    target = self.index.resolve_expr(node.value.func, mod.name, None)
                except Exception:
                    target = None
                if not (target and target[0] == "func"):
                    continue
                # `_connect_decomp_kit("als_native_call", ...)`: the entry the
                # connect binds is named by the call's own string arguments
                said = frozenset(a.value for a in ast.walk(node.value)
                                 if isinstance(a, ast.Constant) and isinstance(a.value, str)
                                 and self.IDENT.match(a.value))
                for t in node.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            bound.setdefault(n.id, []).append(target)
                            self.bound_strings.setdefault(mod.name, {}).setdefault(n.id, set()).update(said)
            self.bound_calls[mod.name] = bound
        self._facts, self._strs = {}, {}

    def methods_of(self, cls):
        if cls.key not in self.methods:
            out = {}
            for sub in cls.node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[sub.name] = self.index.func(cls.mod, sub, cls=cls)
            self.methods[cls.key] = out
        return self.methods[cls.key]

    def facts(self, kind, obj):
        """(definitions it names, attribute/string names) for one definition."""
        key = (kind, obj.key)
        if key in self._facts:
            return self._facts[key]
        defs, names, strs = [], set(), set()
        if kind == "f":
            nodes, scope = [obj.node], obj
        else:
            nodes = [n for n in obj.node.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            nodes += list(obj.node.bases)
            scope = None
        for root in nodes:
            for n in ast.walk(root):
                target = None
                if isinstance(n, ast.Name):
                    names.add(n.id)
                    defs.extend(self.bound_calls.get(obj.mod.name, {}).get(n.id, ()))
                    said = self.bound_strings.get(obj.mod.name, {}).get(n.id, ())
                    names |= set(said)
                    strs |= set(said)
                    try:
                        target = self.index.resolve_expr(n, obj.mod.name, scope)
                    except Exception:
                        target = None
                elif isinstance(n, ast.Attribute):
                    names.add(n.attr)
                    try:
                        target = self.index.resolve_expr(n, obj.mod.name, scope)
                    except Exception:
                        target = None
                elif isinstance(n, ast.Constant) and isinstance(n.value, str) and self.IDENT.match(n.value):
                    names.add(n.value)
                    strs.add(n.value)
                elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add) and \
                        isinstance(n.right, ast.Constant) and isinstance(n.right.value, str) and \
                        self.IDENT.match(n.right.value) and n.right.value.startswith("_"):
                    # `name + "_rowmajor_export"`: an entry named by composition
                    names.add(self.SUFFIX + n.right.value)
                if target and target[0] in ("func", "class"):
                    defs.append(target)
                elif isinstance(n, ast.Name):
                    names |= self.table_names(obj.mod, n.id)
        self._facts[key] = (defs, frozenset(names))
        self._strs[key] = frozenset(strs)
        return self._facts[key]

    def table_names(self, mod, name, depth=0):
        """The identifier strings and names of a module-level table a
        definition reads by name (`_ENCODERS["<f4"] = ("encode_labels_f32", ...)`
        in _labels.py, dispatched with getattr), followed through one import."""
        if depth > 2:
            return frozenset()
        out = set()
        value = mod.aliases.get(name)
        if value is not None and not isinstance(value, (ast.Name, ast.Attribute)):
            for n in ast.walk(value):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and self.IDENT.match(n.value):
                    out.add(n.value)
                elif isinstance(n, ast.Name):
                    out.add(n.id)
        elif name in mod.imports and mod.imports[name][0] == "from":
            other = self.index.mod(mod.imports[name][1])
            if other is not None:
                out |= self.table_names(other, mod.imports[name][2], depth + 1)
        return frozenset(out)

    def closure(self, seeds):
        reached = ReachedSet(set(seeds) | HARNESS_OPS)
        strings = set()
        seen, mods, classes = set(), set(), {}
        queue = []
        for name in seeds:
            for mod in ("mojolearn",):
                try:
                    t = self.index.resolve_name(mod, name)
                except Exception:
                    t = None
                if t and t[0] in ("func", "class"):
                    queue.append(t)
                elif name[:1].isupper():
                    # `ml.ops.Session`, `ml.models.CausalLM`: a class reached
                    # through a subpackage attribute chain
                    queue.extend(x for x in self.by_name.get(name, ()) if x[0] == "class")
                elif name not in self.OPS_NAMES and not name.startswith("__"):
                    # `ml.hdbscan.approximate_predict`: a function reached the
                    # same way, when few modules define the name
                    funcs = [x for x in self.by_name.get(name, ()) if x[0] == "func"]
                    if 0 < len(funcs) <= 3:
                        queue.extend(funcs)
        pending = defaultdict(list)          # method name -> [Func] of reached classes not yet followed
        while queue:
            kind, obj = queue.pop()
            k = ("f" if kind == "func" else "c", obj.key)
            if k in seen:
                continue
            seen.add(k)
            if obj.mod.name not in mods:
                mods.add(obj.mod.name)
                queue.extend(("func", f) for f in self.top.get(obj.mod.name, ()))
            defs, names = self.facts(k[0], obj)
            strings |= self._strs.get(k, frozenset())
            fresh = names - reached
            reached |= names
            queue.extend(defs)
            if kind == "class":
                classes[obj.key] = obj
                for base in self.index.bases(obj) if hasattr(self.index, "bases") else ():
                    if hasattr(base, "node"):
                        queue.append(("class", base))
                for mname, func in self.methods_of(obj).items():
                    if mname in reached or (mname.startswith("__") and mname.endswith("__")):
                        queue.append(("func", func))
                    else:
                        pending[mname].append(func)
            for name in fresh:
                for func in pending.pop(name, ()):
                    queue.append(("func", func))
        reached.strings = frozenset(strings)
        reached.classes = frozenset(c.name for c in classes.values())
        return reached

    @classmethod
    def names_entry(cls, reached, entry, depth=0):
        """`entry` is reached by name, or as a reached name plus one or two
        suffixes a reached definition composes (`name + "_rowmajor"`)."""
        if entry in reached:
            return True
        if depth >= 2:
            return False
        key = id(reached)
        if key not in cls._suffixes:
            cls._suffixes[key] = [n[len(cls.SUFFIX):] for n in reached if n.startswith(cls.SUFFIX)]
        for suffix in cls._suffixes[key]:
            if True:
                if entry.endswith(suffix) and len(entry) > len(suffix) and \
                        cls.names_entry(reached, entry[:-len(suffix)], depth + 1):
                    return True
        return False


def lane_names(ls, ib):
    """lane -> the names its own code touches: the lane function, its lambdas
    and closures, and the helpers it calls transitively, EXCEPT the harness's
    shared helpers (`_fit`, `_h`, the batch drivers: any helper more than a
    quarter of the lanes reach), whose names would hand every lane every
    model the harness can name."""
    per_lane = {}
    count = defaultdict(int)
    for lane, fn in ib.LANES.items():
        funcs = []
        try:
            ls._code_names(fn, fn.__globals__, funcs=funcs)
        except Exception:
            pass
        ids = {id(f) for f in funcs[1:]}
        per_lane[lane] = ids
        for i in ids:
            count[i] += 1
    universal = {i for i, c in count.items() if c > len(ib.LANES) // 4}
    out = {}
    for lane, fn in ib.LANES.items():
        names, seen, stack = set(), set(), [fn]
        while stack:
            f = stack.pop()
            if id(f) in seen:
                continue
            seen.add(id(f))
            codes = [f.__code__]
            for cell in (getattr(f, "__closure__", None) or ()):
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if callable(val) and hasattr(val, "__code__") and id(val) not in universal:
                    stack.append(val)
            while codes:
                code = codes.pop()
                names |= set(code.co_names)
                for const in code.co_consts:
                    if hasattr(const, "co_names"):
                        codes.append(const)
                    elif isinstance(const, str) and NameClosure.IDENT.match(const):
                        names.add(const)
            g = getattr(f, "__globals__", {})
            for name in list(names):
                helper = g.get(name)
                if callable(helper) and hasattr(helper, "__code__") and id(helper) not in universal \
                        and id(helper) not in seen:
                    stack.append(helper)
        for registry in PROPERTY_REGISTRIES:
            spec = getattr(ib, registry, {}).get(lane)
            if spec is not None:
                names |= spec_names(spec)
        out[lane] = names
    return out


#: The harness's per-lane property declarations: the batch, gradient, scale,
#: ragged, step-versus-full and replay probes run calls of their own on the
#: lane's fitted estimator (`_batch_decl(_rows_calls("decode_bytes_batch"...))`).
PROPERTY_REGISTRIES = ("BATCH", "BATCHGRAD", "BATCHSCALE", "RAGGED", "STEPFULL", "RLPAIR")


def spec_names(spec, depth=0, seen=None):
    """The names a property declaration's calls touch: its method-name
    strings and the code of its callables, recursively."""
    seen = set() if seen is None else seen
    if depth > 6 or id(spec) in seen:
        return set()
    seen.add(id(spec))
    out = set()
    if isinstance(spec, str):
        return {spec} if NameClosure.IDENT.match(spec) else set()
    if isinstance(spec, dict):
        for v in spec.values():
            out |= spec_names(v, depth + 1, seen)
        return out
    if isinstance(spec, (list, tuple, set, frozenset)):
        for v in spec:
            out |= spec_names(v, depth + 1, seen)
        return out
    code = getattr(spec, "__code__", None)
    if code is not None:
        codes = [code]
        while codes:
            c = codes.pop()
            out |= set(c.co_names)
            codes.extend(k for k in c.co_consts if hasattr(k, "co_names"))
            out |= {k for k in c.co_consts if isinstance(k, str) and NameClosure.IDENT.match(k)}
        for cell in (getattr(spec, "__closure__", None) or ()):
            try:
                out |= spec_names(cell.cell_contents, depth + 1, seen)
            except ValueError:
                pass
        return out
    if hasattr(spec, "__dict__") and not isinstance(spec, type):
        for v in vars(spec).values():
            out |= spec_names(v, depth + 1, seen)
    return out


#: lane -> the entries its typed walk and its own code reach, before the name
#: closure widens them: the owner entries a generic dispatch branch is credited through
TYPED_ENTRIES = {}


def lane_reach(ls, walker, closer=None):
    """lane -> (set of (public key, op) roots, set of entry names reached, name closure)."""
    ib = ls.identity_break()
    _, why_map = ls.lane_sources()
    own = lane_names(ls, ib)
    out = {}
    for lane in ib.LANES:
        names = own.get(lane, set())
        symbols = set(why_map.get(lane, {}).get("symbols", ())) | names
        roots, entries = set(), set()
        for sym in symbols:
            for kind, obj in walker.by_name.get(sym, ()):
                if kind == "func":
                    funcs, ent = walker.walk(obj, None)
                    roots.add((obj.key, "call"))
                    entries |= ent
                else:
                    for op, func in walker.ops(obj, names):
                        funcs, ent = walker.walk(func, obj)
                        roots.add((obj.key, op))
                        entries |= ent
        entries |= names                        # a lane that calls a binding entry by name
        TYPED_ENTRIES[lane] = frozenset(entries)
        reached = closer.closure(symbols) if closer is not None else set()
        entries |= reached
        out[lane] = (roots, entries, reached)
    return out


# ------------------------------------------------------------ native owners

#: The variables a native owner dispatches on: `if operation == "croston_fit":`
#: (bindings/xseq_forecast_native.mojo), `if family == "FastICA":` then
#: `if operation == "fit":` (bindings/decomp_native.mojo), `kind == "..."`.
DISPATCH_VARS = ("operation", "op", "family", "kind", "kind_name", "method", "command", "name")
_COND_RE = re.compile(r'\b(?:%s)\s*==\s*"([A-Za-z_][A-Za-z0-9_]*)"' % "|".join(DISPATCH_VARS))
_OP_RE = re.compile(r'\b(?:operation|op)\s*==\s*"([A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+)"')
_LIT_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"')
_PREFIX_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*_)"\s*\+')
_CALLEE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*[\[(]")
#: How far a branch's callees are followed for the entry names they hold: the
#: branch's handler (`_croston_fit`) and the helpers it hands the names to.
OWNER_DEPTH = 4
#: An owner file held by more than this many owner entries' closures is shared
#: plumbing (an owner entry's GPU binding and host twin, and one alias, hold
#: their own owner file)
SHARED_OWNER_FILE = 3


class OwnerDispatch:
    """THE PUBLIC RUNTIME MIGRATION'S REACH (2026-10-24). A migrated public
    method is one native call, `_XSEQ(self, "croston_fit", args, context)`;
    the Python walk stops at the owner entry (`xseq_native_call`), and the
    owner, in Mojo, picks the handler by the operation (and family) string and
    calls the compiled entries by NAME (`entries["croston"]`,
    `decomp_entries(..., Python.tuple("x_decomp_ica_fit"), ...)`). So a lane
    that reaches the dispatch strings reaches the entries that branch names.

    Read from the source: every `if`/`elif` on a dispatch variable compared
    with a string literal opens a branch; the branch's lines, and the
    definitions they call (to OWNER_DEPTH), contribute every string literal
    that is a registered entry name. A lane reaches those entries when, for
    every enclosing condition, one of its alternatives (`a == "x" or a == "y"`)
    is among the lane's reached names. A branch whose conditions are all
    GENERIC (a value several owners dispatch on, `fit`, `transform`) credits
    no lane: `fit` alone does not say which owner ran."""

    def __init__(self, ls, exports, methods=frozenset()):
        self.ls, self.exports = ls, frozenset(exports)
        self.graph = ls._MojoDefGraph()
        self._memo, self._prefix_memo, self.op_values = {}, {}, set()
        self.records = []                  # (conds: tuple[frozenset], entries: frozenset, file)
        files = [p for p in git("ls-files", "*.mojo").split() if p]
        seen_in = defaultdict(set)
        for path in files:
            try:
                text = ls._read(path)
            except OSError:
                continue
            if "PythonObject" not in text or '== "' not in text:
                continue
            units = ls._mojo_units(text)
            if units is None:
                continue
            for unit, utext in ls._mojo_unit_texts(units).items():
                for conds, ents, lits in self._branches(path, utext):
                    self.records.append((conds, ents, path, lits))
                    for alt in conds:
                        for v in alt:
                            seen_in[v].add(path)
        # GENERIC: a value several owners dispatch on (`fit`, `transform`), a
        # plain lowercase word (`target`, `column`: any lane's code holds one)
        # or a short code (`M`, `O`). Only a specific value (`croston_fit`,
        # `FastICA`) says which owner branch a lane reaches. A value that is a
        # method name of a package class (`fit_predict`, `predict_proba`) is
        # generic too: every estimator lane names it.
        self.generic = frozenset(v for v, fs in seen_in.items()
                                 if len(fs) >= 3 or len(v) < 4 or ("_" not in v and v.islower())
                                 or v in methods)
        # owner entry -> the owner file its registration's impl is in: a
        # branch on generic values alone (`operation == "fit"` in
        # bindings/ipca_native.mojo) is credited to a lane that reaches the
        # owner entry itself (`ipca_native_call`)
        self.owner_files = defaultdict(set)
        record_files = {r[2] for r in self.records}
        # a branch that names another branch's dispatch value relays to it
        # (`msel_split` hands its lazy iterator the owner operation
        # "family_msel_split_loo_at", bindings/msel_split.mojo): the value
        # joins the lane's reached names. Generic values never relay.
        values = set(self.op_values) - self.generic
        self.records = [(c, e, p, frozenset(l & values)) for c, e, p, l in self.records]
        for src in git("ls-files", "bindings/*.mojo", "bindings/**/*.mojo").split():
            regs = registrations(ls._read(src))
            for export, (impl, params) in regs.items():
                if not export.endswith("_call"):
                    continue
                # the owner's own entry file: the one its registered impl is in
                # (`ipca_c` and its `operation == "fit"` branch, bindings/ipca_native.mojo);
                # an umbrella owner's other files are reached by their own values
                roots = set(self.graph.resolve(src, impl))
                self.owner_files[export] |= {n[1] for n in roots} & record_files
        # a file in many owners' closures is shared plumbing (the codec, the
        # process pool), not the owner: reaching a branch there says nothing
        # about which owner ran, so it credits no owner entry
        holders = defaultdict(set)
        for export, fs in self.owner_files.items():
            for f in fs:
                holders[f].add(export)
        self.shared_files = frozenset(f for f, es in holders.items() if len(es) > SHARED_OWNER_FILE)

    def _strings(self, node, depth):
        """(literals, prefixes) over `node` and what it calls, to `depth`
        calls deep: every identifier-shaped string literal, and every literal a
        name is composed from (`near["xn_" + name]`, bindings/kpca_native.mojo).
        A reached type's method is followed when a reached definition names it
        (a kit's `self.fn("x_decomp_dev_gemm")`, bindings/mani_kit.mojo). Only
        the Python-facing Mojo under bindings/ is read: a kernel names no entry."""
        key = (node, depth)
        if key in self._memo:
            return self._memo[key]
        lits, pres, mentioned = set(), set(), set()
        seen, frontier, waiting = {node}, [node], defaultdict(list)
        for level in range(depth + 1):
            nxt = []
            for n in frontier:
                try:
                    text = self.graph.text(n) or ""
                except Exception:
                    text = ""
                lits |= set(_LIT_RE.findall(text))
                pres |= set(_PREFIX_RE.findall(text))
                fresh = set(self.graph.idents(n)) - mentioned
                mentioned |= fresh
                if level == depth:
                    continue
                cands = list(self.graph.targets(n))
                for ident in fresh:
                    cands.extend(waiting.pop(ident, ()))
                for t in cands:
                    if t in seen or not t[1].startswith("bindings/") or \
                            os.path.basename(t[1]).startswith("_mojolearn"):
                        continue
                    seen.add(t)
                    nxt.append(t)
                    if t[0] == "t":
                        units = self.graph.units(t[1]) or {}
                        for meth in units.get("methods", {}).get(t[2], {}):
                            m = ("m", t[1], t[2], meth)
                            if m in seen:
                                continue
                            if meth in mentioned:
                                seen.add(m)
                                nxt.append(m)
                            else:
                                waiting[meth].append(m)
            frontier = nxt
        self._memo[key] = (frozenset(lits), frozenset(pres))
        return self._memo[key]

    def _named(self, lits, pres):
        out = set(lits) & self.exports
        for p in pres:
            for e in self._by_prefix(p):
                if e[len(p):] in lits:
                    out.add(e)
        return out

    def _by_prefix(self, p):
        if p not in self._prefix_memo:
            self._prefix_memo[p] = [e for e in self.exports if e.startswith(p) and len(e) > len(p)]
        return self._prefix_memo[p]

    def _line_entries(self, path, line):
        lits, pres = set(_LIT_RE.findall(line)), set(_PREFIX_RE.findall(line))
        for ident in set(_CALLEE_RE.findall(line)):
            try:
                nodes = self.graph.resolve(path, ident)
            except Exception:
                nodes = ()
            for n in nodes:
                a, b = self._strings(n, OWNER_DEPTH - 1)
                lits |= a
                pres |= b
        return self._named(lits, pres), lits

    def _branches(self, path, text):
        stack = []                         # [(indent, frozenset alternatives | None)]
        acc = defaultdict(set)
        lit_acc = defaultdict(set)
        for raw in text.splitlines():
            body = raw.rstrip()
            cut = body.find("#")
            if cut >= 0 and body[:cut].count('"') % 2 == 0:
                body = body[:cut].rstrip()
            stripped = body.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(body) - len(body.lstrip())
            while stack and stack[-1][0] >= indent:
                stack.pop()
            head = stripped.split(" ", 1)[0]
            if head in ("if", "elif") and stripped.endswith(":"):
                # `a == "x" or a == "y"` is one requirement with two
                # alternatives; `... and b == "z"` adds a second requirement
                self.op_values |= set(_OP_RE.findall(stripped))
                parts = [frozenset(_COND_RE.findall(p)) for p in stripped[len(head):-1].split(" and ")]
                parts = [p for p in parts if p]
                stack.append((indent, tuple(parts) or None))
                continue
            if head == "else:" or stripped == "else:":
                stack.append((indent, None))
                continue
            conds = tuple(alt for _, c in stack if c for alt in c)
            if not conds:
                continue
            ents, lits = self._line_entries(path, stripped)
            if ents or lits:
                acc[conds] |= ents
                lit_acc[conds] |= lits
        return [(c, frozenset(acc[c]), frozenset(lit_acc[c])) for c in acc]

    def reached(self, names, owners=None):
        """(entry names, owner files): what the owners' branches reach for a
        lane whose reached names are `names`, and the files of those branches."""
        out, files = set(), set()
        owned = set()
        for export, fs in self.owner_files.items():
            if export in (owners if owners is not None else names):
                owned |= fs - self.shared_files
        names, done = set(names), set()
        grew = True
        while grew:
            grew = False
            for k, (conds, ents, path, relay) in enumerate(self.records):
                if k in done:
                    continue
                if all(alt & names for alt in conds) and (
                        path in owned or any((alt & names) - self.generic for alt in conds)):
                    done.add(k)
                    out |= ents
                    files.add(path)
                    if relay - names:
                        names |= relay
                        grew = True
        return out, files


def owner_reach(ls, reach):
    """`reach` with every lane's entries widened by the native owners' dispatch."""
    exports = set()
    for src in git("ls-files", "bindings/*.mojo", "bindings/**/*.mojo").split():
        exports |= set(registrations(ls._read(src)))
    methods = set()
    for path in git("ls-files", "python/mojolearn").split():
        if not path.endswith(".py") or "/tests/" in path:
            continue
        try:
            tree = ast.parse(open(os.path.join(ROOT, path)).read())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods |= {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    od = OwnerDispatch(ls, exports, frozenset(methods))
    # an owner entry that most lanes' name closures hold (an umbrella owner the
    # shared mixins name: `classical_native_call`) credits its generic branches
    # only to the lanes whose typed walk or own code reaches it; any other owner
    # entry (`ipca_native_call`, named by its class's bridge connect) credits
    # them to every lane whose closure names it
    held = defaultdict(int)
    for lane, (_, entries, _) in reach.items():
        for export in od.owner_files:
            if export in entries:
                held[export] += 1
    umbrella = {e for e, n in held.items() if n > len(reach) // 4}
    out = {}
    for lane, (roots, entries, reached) in reach.items():
        owners = (set(entries) - umbrella) | (set(TYPED_ENTRIES.get(lane, ())) & umbrella)
        dispatch = set(getattr(reached, "strings", ())) | set(getattr(reached, "classes", ())) | \
            (set(entries) - set(reached))
        extra, files = od.reached(frozenset(dispatch), frozenset(owners))
        out[lane] = (roots, entries | extra, reached)
        OWNER_FILES[lane] = frozenset(f for f in files if "native" in os.path.basename(f)
                                      and f not in od.shared_files)
    return out, od


#: lane -> the native owner files (`*native*.mojo`) whose dispatch branches the
#: lane reaches; a registration whose Mojo closure holds one of them (the owner
#: entry itself: `xseq_native_call`, `decomp_native_call`) is reached by the lane.
OWNER_FILES = {}


def python_rows(ls, walker, reach, base, whole_surface=False):
    pri = walker.pri
    index = walker.index
    changed = changed_python_lines(base)
    root_lanes = defaultdict(set)
    for lane, (roots, _, _) in reach.items():
        for r in roots:
            root_lanes[r].add(lane)
    rows = []

    def touched(func):
        rel = os.path.relpath(str(func.mod.path), ROOT)
        lines = changed.get(rel)
        if not lines:
            return False
        end = getattr(func.node, "end_lineno", func.node.lineno)
        return any(func.node.lineno <= ln <= end for ln in lines)

    done = set()
    for canonical, origin, aliases in pri.public_names(pri.REPO):
        module, _, name = origin.rpartition(".")
        target = index.resolve_name(module, name) if module else None
        if target is None or target[0] in ("module", "const"):
            continue
        kind, obj = target
        if kind == "func":
            ops = [("call", obj, None)]
        else:
            ops = []
            for op in pri.OPS:
                found = index.lookup(obj, op)
                if found and found[0] == "func" and not pri._refusal(found[1].node):
                    ops.append((op, found[1], obj))
        for op, func, cls in ops:
            key = (func.key, cls.key if cls else None)
            if key in done:
                continue
            done.add(key)
            funcs, entries = walker.walk(func, cls)
            path_changed = [f for f in funcs if touched(f)]
            if not path_changed and not whole_surface:
                continue
            owner = cls.key if cls else func.key
            lanes = set(root_lanes.get((owner, op), ()))
            if cls is not None:
                # a subclass the lane names reaches the same inherited method
                for (k, o), ls_ in root_lanes.items():
                    if o == op and k != owner:
                        sub = index.classes.get(k)
                        if sub is not None and cls in index.mro(sub):
                            lanes |= ls_
            short = cls.name if cls is not None else func.node.name
            for lane, (_, _, reached) in reach.items():
                if short in reached and (cls is None or op in reached):
                    lanes.add(lane)
            rel = os.path.relpath(str(func.mod.path), ROOT)
            rows.append(dict(public=canonical, op=op, where=f"{rel}:{func.node.lineno}", own=touched(func),
                             trivial=trivial_method(func.node),
                             helpers_changed=[f"{os.path.relpath(str(f.mod.path), ROOT)}:{f.qual}"
                                              for f in path_changed[:3]],
                             n_helpers_changed=len(path_changed), entries=sorted(entries),
                             lanes=sorted(lanes)))
    rows.sort(key=lambda r: (r["public"], r["op"]))
    return rows


def endpoint_rows(walker, reach):
    """Map the complete existing endpoint census onto actual lane call roots.

    Mapping is source reach, never runtime success. Do not credit a missing
    public call merely because another method calls the same native export.
    Metadata/lifetime-only endpoints have no numerical identity proposition;
    the separate runtime checker still owns their behavior.
    """
    pri, index = walker.pri, walker.index
    census = pri.endpoint_inventory(pri.REPO)
    origins = {}
    for canonical, origin, aliases in pri.public_names(pri.REPO, include_non_algorithm=True):
        for name in (canonical, *aliases):
            origins[name] = origin
    by_root = defaultdict(set)
    for lane, (roots, _, _) in reach.items():
        for root in roots:
            by_root[root].add(lane)
    metadata = {"__repr__", "__str__", "__sklearn_tags__", "__sklearn_is_fitted__",
                "__init_subclass__", "numeric_mode_used", "vendor_used"}
    lifetime = {"__del__", "close", "__enter__", "__exit__", "free"}
    rows = []
    for endpoint in census['endpoints']:
        public, method = endpoint['public'], endpoint['method']
        origin = origins.get(public, public)
        module, _, name = origin.rpartition('.')
        target = index.resolve_name(module, name) if module else None
        op = method.split(':', 1)[0]
        lanes = set()
        if target and target[0] in ('class', 'func'):
            obj = target[1]
            lanes.update(by_root.get((obj.key, op), ()))
            # Only explicit calls on this class count. Inherited aliases may
            # share an implementation; they still retain separate API rows.
            if target[0] == 'class':
                found = index.lookup(obj, op)
                if found and found[0] == 'func':
                    for lane, (roots, _, _) in reach.items():
                        if (obj.key, op) in roots:
                            lanes.add(lane)
        adapter = '.adapters.' in public or '.adapters.' in origin
        reason = None
        if adapter:
            reason = 'adapter boundary: installed-wheel smoke; arithmetic shares the existing kernel lanes'
        elif op in metadata:
            reason = 'metadata only; no numerical identity output'
        elif op in lifetime:
            reason = 'lifetime/context management; no numerical identity output'
        elif public == 'mojolearn.ops.Session' and op in ('profile_reset', 'profile_report'):
            reason = 'launch profiling diagnostics; no numerical identity output'
        elif endpoint.get('always_refuses'):
            reason = 'explicit unsupported-operation refusal; no numerical identity output'
        rows.append(dict(endpoint=endpoint['endpoint'], public=public, op=method,
                         kind=endpoint['kind'], where=endpoint['definition'],
                         aliases=endpoint['aliases'], lanes=sorted(lanes),
                         status='not_numerical' if reason else 'mapped' if lanes else 'unmapped',
                         reason=reason, native_entries=endpoint['native_entries'],
                         execution='not assessed'))
    # Resolve direct receivers in the actual fixtures as well as the older
    # estimator walk (which cannot infer ``with ops.Session() as s``).
    import lane_select
    from verification_endpoint_calls import lane_calls, source_calls
    direct = lane_calls(lane_select.identity_break(), rows)
    supplemental = defaultdict(set)
    from light_verifier_reference import load
    for module in load('_verification_sources').SUPPLEMENTAL_CHECKS:
        path = os.path.join(ROOT, 'python', 'mojolearn', module + '.py')
        if not os.path.isfile(path):
            continue
        tree = ast.parse(open(path).read())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        roots = ('verify',) if module.endswith('_program') else ()
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPECTED_CASES' for t in n.targets):
                roots = ast.literal_eval(n.value)
        for endpoint, cases in source_calls(functions, rows, roots).items():
            supplemental[endpoint].update(module + ':' + case for case in cases)
    for row in rows:
        row['lanes'] = sorted(set(row['lanes']) | direct.get(row['endpoint'], set()))
        row['checks'] = sorted(supplemental.get(row['endpoint'], ()))
        if row['status'] != 'not_numerical':
            row['status'] = 'mapped' if row['lanes'] or row['checks'] else 'unmapped'
    return rows


def trivial_method(node):
    """True for a method with no work at all: its body, past the docstring, is
    `return self`, `return`, `pass` or `return self.<attribute>` (a stateless
    layer's `fit`)."""
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 1:
        return False
    st = body[0]
    if isinstance(st, ast.Pass):
        return True
    if isinstance(st, ast.Return):
        v = st.value
        return v is None or (isinstance(v, ast.Name) and v.id == "self") or (
            isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "self")
    return False


def py_triage(row, native_by_entry):
    """(status, lanes, why) for one public method row.

    THE PYTHON-SHELL RULE (2026-10-24). A public method is API shell: it checks
    arguments and routes to a native call. It counts as covered when a lane
    reaches it, or, failing that, when every native entry its walked call path
    reaches is itself covered (the shell adds no arithmetic of its own; the
    no-Python-compute fence holds it to that). A method with no native work at
    all (`return self`) is n/a."""
    key = (row["public"], row["op"])
    if key in PY_ANNOTATIONS:
        return PY_ANNOTATIONS[key]
    if row["lanes"]:
        return "covered", row["lanes"], ""
    entries = [e for e in row.get("entries", ()) if e in native_by_entry]
    if entries and all(native_by_entry[e][0] in ("covered", "covered (new lane)") for e in entries):
        lanes = sorted({l for e in entries for l in native_by_entry[e][1]})
        return "covered (via native)", lanes, "every native entry it calls is covered: " + ", ".join(
            f"`{e}`" for e in entries[:3]) + (f" +{len(entries) - 3}" if len(entries) > 3 else "")
    if not row.get("entries") and row.get("trivial"):
        return "n/a", [], "no native work: the method only returns (a stateless layer's fit returns self)"
    return "GAP", [], "no lane reaches it"


def native_status_by_entry(nat):
    """entry name -> (best status, lanes) over its registrations (GPU binding and host twin)."""
    rank = {"covered": 0, "covered (new lane)": 0, "owed": 1, "n/a": 2, "GAP": 3}
    out = {}
    for r in nat:
        st, lanes, _ = triage(r)
        st = "covered (new lane)" if st == "covered" and lanes and set(lanes) <= NEW_LANES else st
        prev = out.get(r["entry"])
        if prev is None or rank.get(st, 9) < rank.get(prev[0], 9):
            out[r["entry"]] = (st, lanes)
        elif rank.get(st, 9) == rank.get(prev[0], 9):
            out[r["entry"]] = (prev[0], sorted(set(prev[1]) | set(lanes)))
    return out


def recorded_lanes():
    try:
        table = json.load(open(os.path.join(ROOT, "python/mojolearn/verify_reference/table.json")))
    except OSError:
        return set()
    out = set()
    for cell, parts in table.get("cells", {}).items():
        lane = cell.rsplit("/", 1)[0]
        for part in parts.values():
            ref = part.get("ref") if isinstance(part, dict) else None
            if isinstance(ref, str) and re.fullmatch(r"[0-9a-f]{16}", ref):
                out.add(lane)
                break
    return out


# ------------------------------------------------------------------ triage

#: Historical lane descriptions, not a reference/admission inventory. Current
#: reference status comes only from the retained table snapshot below.
NEW_LANES = frozenset({"gp-lml-theta", "cross-vendor-fold", "sequence-moe-backward",
                       "sequence-export-roundtrip", "sequence-onnx-import",
                       "transformer-decode-load-state", "safetensors-widen-f16",
                       "sequence-program-save-load", "hf-causal-lm-stream", "ivf-short-fill",
                       "sequence-device-tensor",
                       # lane verify-coverage-owed (2026-10-21)
                       "mamba2-decode-session", "mamba3-decode-session", "training-chunked-lm-head",
                       "training-mlp-sessions", "samba-optimizer-state", "sequence-optimizer-state",
                       "x-huber-outliers", "x-sgd-clf-proba", "x-logistic-cv-proba",
                       "x-decomp-rsvd-rank-deficient",
                       # lane verify-gaps-20261022
                       "saved-model-host-infer-trees", "hf-causal-lm-text", "hf-model-surface",
                       "training-device-cross-entropy", "sequence-layernorm-backward",
                       "x-metrics-classification-report", "x-metrics-permutation-test-score",
                       "classical-float16-sequences",
                       # lane verify-coverage-20261024
                       "transformer-forward-tape", "public-cnn-classifier-predict", "public-cnn-layer-fit",
                       "public-numeric-metadata", "public-randomized-search", "public-residual-dropout",
                       "public-scorer-factories", "public-sparse-encode", "public-theta-variants",
                       "public-validation-learning-curves", "scaler-fit-transform", "multioutput-ridge",
                       "x-ridge-cv-kfold", "neural-trainer-unfused", "svc-gamma-scale",
                       # lane coverage-gap-lanes-20261025
                       "public-set-params"})

#: What each new lane is for: the endpoints only it reaches.
NEW_LANE_PURPOSE = {
    "gp-lml-theta": "GaussianProcessRegressor.log_marginal_likelihood(theta, eval_gradient): `gp_theta_params`, "
                    "`gpr_lml_grad`",
    "cross-vendor-fold": "mojolearn.cross_vendor ordered_fold and fold_pair: `fold_pair_f32`, `cross_vendor_call`",
    "sequence-moe-backward": "MoEBlock.backward: `moe_backward`",
    "sequence-export-roundtrip": "Sequential.export, Sequential.load, predict and predict_proba",
    "sequence-onnx-import": "mojolearn.import_model on an ONNX file: `onnx_import`, the Mojo ONNX reader",
    "transformer-decode-load-state": "TransformerDecodeSession.load_state: `transformer_decode_session_load_state`",
    "safetensors-widen-f16": "float16 checkpoint weights (models.safetensors.widen_f16): `from_f16`, device and host",
    "sequence-program-save-load": "ops.Program.save and Session.load_program (the public `ops.Program` class)",
    "hf-causal-lm-stream": "CausalLM.stream and models.GenerationStream.read, greedy and sampled",
    "ivf-short-fill": "IVFIndex.search with fewer candidates than k: the short fill on the device search and "
                      "the host column (`ivf_out_of_bounds_distance`)",
    "sequence-device-tensor": "resident sequence tensors (`seq_tensor_alloc`, `seq_tensor_upload`, "
                              "`seq_tensor_download`, `seq_tensor_copy`) through RMSprop",
    "mamba2-decode-session": "Mamba2Block's resident decode session: step, sync_state and load_state "
                             "(`mamba2_session_open`, `mamba2_session_export_state`, `mamba2_session_load_state`)",
    "mamba3-decode-session": "Mamba3Block's resident decode session: step, sync_state and load_state "
                             "(`mamba3_session_open`, `mamba3_session_step`, `mamba3_session_export_state`, "
                             "`mamba3_session_load_state`)",
    "training-chunked-lm-head": "training.chunked_lm_head_loss with and without gradients: "
                                "`chunked_lm_head_v2_loss`, `chunked_lm_head_v2_train`",
    "training-mlp-sessions": "training.mlp_inference_sessions: `mlp_sessions`",
    "samba-optimizer-state": "SambaStack state_dict / load_state_dict on an open resident optimizer slot "
                             "(`optimizer_resident_download`, `optimizer_resident_upload`) and "
                             "optimizer.maximize (`optimizer_resident_step_io`)",
    "sequence-optimizer-state": "NAdam and RMSprop state_dict / load_state_dict: `seq_optimizer_export`, "
                                "`seq_optimizer_import`",
    "x-huber-outliers": "HuberRegressor.outliers_ (`py2mojo_rows` ROWS_HUBER_OUT)",
    "x-sgd-clf-proba": "SGDClassifier.predict_proba (`py2mojo_rows` ROWS_SGD_PROBA)",
    "x-logistic-cv-proba": "LogisticRegressionCV.predict_proba (`py2mojo_rows` ROWS_LRCV_PROBA)",
    "x-decomp-rsvd-rank-deficient": "randomized_svd, PCA and TruncatedSVD (randomized) of a rank-deficient matrix: "
                                    "the live-column compaction (`x_decomp_order_small`, `x_decomp_dev_order_small`)",
    "saved-model-host-infer-trees": "HostForest and host_predict on saved Extra Trees and default-engine random "
                                    "forest regressors: `forest_host_et_predict`, `forest_host_rf_predict_reg`",
    "hf-causal-lm-text": "CausalLM.generate_text, stream_text and _generate_resident with a tokenizer from "
                         "`Tokenizer(...)`, `_from_json` and `_from_spelled`: `causal_lm_generate_text`, "
                         "`causal_lm_text_stream`, `causal_lm_resident`, `vocab_json_public`, `vocab_spelled_public`",
    "hf-model-surface": "HFConfig, ModelPlan, TensorInfo, SafetensorsFile._map, Checkpoint._file and "
                        "CausalLM._read_weights: `model_config_*`, `model_plan_init`, `tensor_info_*`, "
                        "`safetensors_map`, `checkpoint_file`, `causal_model_read_weights`",
    "training-device-cross-entropy": "training.cross_entropy, linear_forward and linear_backward on device arrays "
                                     "(GPU columns): `ce_loss_dev`, `linear_forward_dev`, `linear_backward_dev`",
    "sequence-layernorm-backward": "the functional layer_norm_backward",
    "x-metrics-classification-report": "metrics.classification_report, dict and text",
    "x-metrics-permutation-test-score": "model_selection.permutation_test_score, with and without groups",
    "classical-float16-sequences": "LinearRegression, Ridge, StandardScaler, MinMaxScaler and PCA on a list of "
                                   "float16 rows: the classical input door's `sequence_widen_halves` / "
                                   "`sequence_narrow_halves` and the scalers' `scaler_cast`",
    # lane verify-coverage-20261024
    "transformer-forward-tape": "TransformerBlock.forward_with_tape, backward_from_tape and "
                                "TransformerForwardTape.close: `transformer_session_forward_tape`, "
                                "`transformer_session_backward_tape`, `transformer_session_discard_tape`",
    "public-cnn-layer-fit": "Conv1d, Conv2d, BatchNorm1d and BatchNorm2d fit",
    "public-cnn-classifier-predict": "CNNClassifier.predict and predict_proba",
    "public-numeric-metadata": "training.vendor_used, ensemble.catboost_auto_learning_rate and "
                               "johnson_lindenstrauss_min_dim",
    "public-randomized-search": "RandomizedSearchCV fit, predict, score, predict_proba, decision_function, transform",
    "public-residual-dropout": "training.residual_dropout and residual_dropout_backward",
    "public-scorer-factories": "model_selection.check_cv, get_scorer, get_scorer_names, make_scorer",
    "public-sparse-encode": "sparse_encode (threshold, lasso_cd, omp)",
    "public-theta-variants": "DynamicTheta and OptimizedTheta fit and predict",
    "public-validation-learning-curves": "model_selection.validation_curve and learning_curve",
    "scaler-fit-transform": "StandardScaler and MinMaxScaler fit_transform: `standard_fit_transform`, "
                            "`minmax_fit_transform`",
    "multioutput-ridge": "MultiOutputRegressor over Ridge: `ridge_predict_multi`",
    "x-ridge-cv-kfold": "RidgeCV with an integer cv: `x_linear_best_alpha`",
    "svc-gamma-scale": "SVC and SVR with gamma='scale': `scale_gamma_limbs`",
    "neural-trainer-unfused": "ParallelNeuralTrainer with MOJOLEARN_MLP_FUSED=0: `mlp_bias_activation`, "
                              "`mlp_relu_backward`, `mlp_sum_rows`",
}


#: What the arms' run changed (lane run-sabotage-arms, 2026-10-21), rendered under Arms proven.
_PROVEN_FIXES = [
    'Fixed during the run (no lane weakened; every arm whose patch or lanes gained something was rerun and caught; the arms that only stopped naming an unreachable lane keep their run):',
    '', "- `coverage-core-hotpath_impl_rows`: x-prep-user-objects did not move. The patch swapped fold labels 0 and 1 in both `assign_fold_i64` and `count_fold_hits_i64`, a consistent relabeling no output can show; the swap now stays in `assign_fold_i64` only, so TargetEncoder's `cv=<splitter>` check refuses. The patch also faults the new host `bincount2_i32`.", '- `coverage-x_decomp-select_ops`: x-decomp-lstsq-rsvd never reaches the live-column compaction (its sketches are no wider than the rank), so it did not move. New lane `x-decomp-rsvd-rank-deficient` (a rank-4 matrix with an 8-wide sketch; 9 of 9 fixtures STABLE, Metal equal to the host column) catches it, and `coverage-x_decomp-select_dev` names it too.', "- Lanes the arm's fault cannot reach, no longer named: par-cholesky and par-hdbscan on the host-binding arms (`coverage-decomposition-linalg_host_impl`, `coverage-hdbscan-hdbscan_door_host_impl`; both read N/A on the host column, which has no multi-device driver), trees-adaboost-reg on `coverage-core-array_helpers_impl` (AdaBoost.R2 predicts by the weighted median and calls none of the faulted helpers).", "- A real host-column bug: `StratifiedGroupKFold.split` refused on a CPU-only install because the GPU base binding's `bincount2_i32` and `strat_group_plan_i32` had no host twin. The core host binding has both now; x-metrics-splitters on the host column is IDENTICAL to the reference on 9 of 9 fixtures, and `coverage-core-hotpath_impl` and `coverage-core-hotpath_impl_rows` were rerun with the fix in the clean set.", '- `coverage-tokenizer-ranks` also lists `build_tokenizer_fast`: that is the FAST-tier binding, which the verifier refuses, so it was not built; the arm is caught through the host tokenizer.',
    '',
    '- 2026-10-22, the seven native-owner arms (coverage-native-*): run on 551dee1ce over the 42a554521 Metal + host build, all CAUGHT (every cell MOVED, restored equal to clean). `coverage_sabotage.py check` now matches import-path module names to the arm\'s bindings, and the three runs share one option set.',
    '',
]

#: What the 2026-10-24 arms' run decided, per MISSED arm, and the accepted
#: product-check catch (lane verify-coverage-20261024, follow-up).
_ARM_DECISIONS_20261024 = [
    "- `coverage-x_neighbors-host_ops` (fixed arm): connected-components and ocsvm MOVED on the host column; the "
    "nearest-centroid lane cannot reach `op_softmax64`, which is registered only in fast builds (NC_PROBA64). The "
    "softmax edit is dropped, the expected lanes are the two that moved, and `xn_softmax64` is fault n/a (fast tier).",
    "- `coverage-x_cnn-perm_host`, `coverage-x_cnn-perm_device` (fixed arms): x-cnn-gcn MOVED on both columns "
    "(edge-weighted GCN runs both the gather and the transposed weighted view); sage and gnn-options do not take "
    "the weighted transpose. Expected lanes: x-cnn-gcn.",
    "- `coverage-resample-gather` (retired, n/a): identical builds compile `resample_gather_gpu` / "
    "`resample_gather_narrow` out (RESAMPLE_GPU_GATHER is fast or the opt-in C11 define, the narrow gather fast), so "
    "the entries return their not-taken flag and no identical lane can move; the resample fit's arithmetic carries "
    "`coverage-core-resample_impl`. The patch and manifest row are removed; the entries are fault n/a.",
    "- `coverage-svm-scale_gamma` (fixed arm, new fault): svc reaches `scale_gamma_limbs` (bindings/svc_kernel_native.mojo "
    "`_resolve_gamma`), but flipping limb 0's lowest bit is 2^-149 of S1, below the once-rounded gamma. The fault now "
    "feeds a wrong cell to the exact sums (cell 0 replaced by cell 1 on the device, counted twice on the host). "
    "The re-proof was MISSED again: svc uses the package default gamma='auto'. New lane `svc-gamma-scale` "
    "(SVC and SVR with gamma='scale', the path above) is the arm's expected lane; the arm builds the svm bindings only.",
    "- `coverage-core-dense_coo` (fixed arm): x-cluster-spectral-affinities MOVED on the host column. A GPU install "
    "never reaches the base binding's COO scan (the metrics dense entry serves every affinity), so the arm is host "
    "only and the GPU `nonzero_f32_fill` row is fault n/a; spectral-precomputed is not an expected lane.",
    "- `coverage-training-mlp_ops_rng` (split; new lane): the RNG half MOVED (samba-untied-dropout-accum) and is "
    "`coverage-training-neural_rng`. The MLP doors run only on the unfused gradient route a multi-device optimizer "
    "takes (`_mlp_fused_enabled`): new lane `neural-trainer-unfused` runs it with MOJOLEARN_MLP_FUSED=0, and "
    "`coverage-training-mlp_ops` expects it.",
    "- `coverage-core-msel_device_host` (new fault, no crash policy; the re-proof's build failure was the patch "
    "joining a comment onto the launch's argument line, fixed): both columns aborted (exit 134; Metal "
    "IOGPUMetalSuballocatorFree) because the old fault left the split table and its sums disagreeing, which drove "
    "out-of-range indices. A crash is not counted as detection: it is not a verifier report and its cause is "
    "memory corruption, not the answer. The fault now yields consistent wrong answers (two gathered rows swapped, "
    "one test group moved to train with its sums, every group rotated to the next fold with its sizes).",
    "- `x-prep-user-objects` (lane revised, 5b1606ab1 AMD record REFUSED on all nine fixtures): LDA/QDA now refuse a "
    "covariance_estimator object at fit (no Python callback in the runtime). The lane keeps the user objects the "
    "library accepts (TargetEncoder splitter, pairs and categories; SimpleImputer is strategy='median' since 2026-10-25, a callable strategy is refused), holds the two "
    "LDA and the QDA refusals as a `refusals` part, and records the native configuration instead (LDA shrinkage=0.25, "
    "QDA reg_param=0.25). Its cells and the arms that expect it (coverage-core-hotpath_impl_rows, "
    "coverage-core-count-fold-hits) are re-recorded / re-proved.",
    "- `coverage-x_cnn-device_ops` (CAUGHT, accepted): the catch is the lane's own product check "
    "(tools/identity_lanes/public_cnn_methods.py:39, predict must equal classes_[argmax(predict_proba)]), the "
    "documented contract of CNNClassifier.predict; that refusal counts as detection.",
]


def _proven_arms():
    """The rows of tools/identity_lanes/coverage_sabotage_proven.tsv (the arms' run), or []."""
    path = os.path.join(ROOT, "tools", "identity_lanes", "coverage_sabotage_proven.tsv")
    try:
        with open(path, encoding="utf-8") as fh:
            return [line.rstrip("\n").split("\t") for line in fh
                    if line.strip() and not line.startswith("#")]
    except OSError:
        return []

_BYTE_SESSION = ("covered", ["byte-lm", "byte-lm-resident"],
                 "the CPU column routes _mojolearn_byte_lm through _byte_lm_trainer_host "
                 "(host_surface ADAPTED_MODULES) to ByteHostSession and the shared native descriptor "
                 "table for every SmallByteLanguageModelTrainer step; on a GPU install "
                 "byte_lm_trainer_step runs the native GPU session owner")
# lane verify-coverage-owed (2026-10-21): the rows the migration lanes left
# owed, each checked in the source after they landed.
_MAMBA_SESSION = ("bindings/mamba_public_native.mojo composes `<kind> + \"_session_<op>\"` and calls the "
                  "binding entry from Mojo (Mamba2Block._decode_session / Mamba3Block._decode_session -> "
                  "mamba_native_call 'session_open', 'session_step', 'session_sync', 'session_load')")
_SAMBA_OWNER = ("the native Samba owner (bindings/samba_*_native.mojo) is handed the entry at import "
                "(_native_samba_bridge.connect) and calls it by name")
_FAST_LINEAR = "fast tier only (Apple): registered for OLS_FAST_NORMAL_EQ / RIDGE_FAST_RESIDENT, which " \
               "need numeric_mode fast on an Apple GPU; the verifier refuses a fast process"
_HOST_SLOT = ("no live caller on a CPU install: the optimizer door moves its slots through its own "
              "kernels; the separate close wrapper is live and annotated below. SambaStack opens no slot on "
              "the host binding, which exports no optimizer_resident_step (_samba_open_optimizer)")

#: entry -> (status, lanes, why), read by hand where the static reach cannot
#: see the call (a worker process, a native owner handed the entry, a class
#: adapter) or where no lane can apply. Each one was checked in the source.
ANNOTATIONS = {
    **{e: _BYTE_SESSION for e in ("byte_lm_session_eval", "byte_lm_session_export_gradients",
                                  "byte_lm_session_export_state", "byte_lm_session_logits",
                                  "byte_lm_session_next_bytes", "byte_lm_session_open",
                                  "byte_lm_session_rollback", "byte_lm_session_step")},
    "knn_classify_neighbors": ("covered", ["par-reference-knn"],
                               "run by bindings/parallel_neighbor_worker.mojo reference_vote; "
                               "par-* lanes run on two-device boxes"),
    "knn_regress_neighbors": ("covered", ["par-reference-knn-reg"],
                              "run by bindings/parallel_neighbor_worker.mojo reference_vote"),
    "fold_pair_f32": ("covered", ["cross-vendor-fold"],
                      "mojolearn.cross_vendor hands it to the native fold at import "
                      "(_buffer_native('fold_pair_f32')); every ordered_fold and fold_pair runs it"),
    "ivf_flat_partial_search": ("covered", ["par-ivf"], "run by bindings/parallel_neighbor_worker.mojo ivf_search"),
    "ivf_finalize_distances": ("covered", ["par-ivf"],
                               "called by the native distributed IVF owner (bindings/parallel_shard_native.mojo "
                               "`ivf_search`) on a cosine index"),
    "pca_fit": ("covered", ["pca", "pca-full-whiten", "pca-whiten"],
                "the native PCA owner is handed it at connect (_native_pca_bridge.connect)"),
    "x_decomp_pca_mle_rank": ("covered", ["x-decomp-pca-randomized"],
                              "PCA(n_components='mle') through the native PCA owner (_native_pca_bridge)"),
    "x_decomp_dev_pca_mle_rank": ("covered", ["x-decomp-pca-randomized"],
                                  "PCA(n_components='mle') through the native PCA owner (_native_pca_bridge)"),
    **{e: ("n/a", [], "fast tier only (Apple): the verifier runs identical and refuses a fast process")
       for e in ("fast_from_bf16", "fast_matmul_bf16", "fast_to_bf16", "public_fast_from_bf16",
                 "public_fast_to_bf16")},
    "ops_shape": ("n/a", [], "the verifier's own native-session boundary probe (_verify_graph)"),
    "device_plan_log": ("n/a", [], "read only by the retired `verify --par` two-device placement witness "
                        "(_verify_par PoolWitness, removed 2026-10-31): device groups, no data value"),
    "covariance": ("n/a", [], "Apple FAST quality probe binding (tools/scoped_gemm_quality.py)"),
    **{e: ("covered", ["mamba2-decode-session"], _MAMBA_SESSION)
       for e in ("mamba2_session_export_state", "mamba2_session_load_state", "mamba2_session_open")},
    **{e: ("covered", ["mamba3-decode-session"], _MAMBA_SESSION)
       for e in ("mamba3_session_export_state", "mamba3_session_load_state", "mamba3_session_open",
                 "mamba3_session_step")},
    "samba_head_loss": ("covered", ["samba-untied-dropout-accum"],
                        _SAMBA_OWNER + ": samba_gradients runs the fused head (`fused`) whenever the "
                        "resident Samba step does not take the step (dropout or accumulation) and the "
                        "chunked head is off; the fused tail that would replace it is Apple FAST only "
                        "(AFN_SAMBA_FUSE). The row is the GPU binding's"),
    "neural_rng": ("covered", ["samba", "samba-untied-dropout-accum"],
                   _SAMBA_OWNER + ": samba_construct draws every random weight from the stack's Generator "
                   "through it and _samba_dropout draws the masks (samba-untied-dropout-accum)"),
    "lr_schedule_exact": ("covered", ["samba-untied-dropout-accum"],
                          _SAMBA_OWNER + " (`exact_schedule`): samba_schedule_rate evaluates the stack's "
                          "WarmupCosineLR at every step through it"),
    "optimizer_maximize_dev": ("n/a", [], "a capability probe returning a build constant; its one caller "
                                          "(samba_train_step) tests its presence only, and the maximized "
                                          "step it selects is optimizer_resident_step_io"),
    "optimizer_resident_step_io": ("covered", ["samba-optimizer-state"],
                                   _SAMBA_OWNER + ": samba_train_step steps a resident slot through it "
                                   "when the optimizer maximizes (or holds device parameters); the lane "
                                   "sets optimizer.maximize on a stack with dropout, so the resident Samba "
                                   "step does not take it"),
    "linear_gram_fit": ("covered", ["ridge", "ridge-no-intercept"],
                        "the native classical owner (bindings/classical_ridge.mojo) is handed it at connect "
                        "(_native_classical_bridge); Ridge.fit on more than one column without weights "
                        "takes it whenever linear_gram_solve_default (LINEAR_GRAM_SOLVE, on in identical)"),
    "ols_predict": ("covered", ["ols", "ridge", "ols-no-intercept", "ridge-no-intercept"],
                    "the native classical owner's classical_linear_predict calls it for every "
                    "LinearRegression and Ridge predict (_native_classical_bridge connect)"),
    "x_decomp_ols_tsqr_r": ("covered", ["ols"],
                            "the native classical owner (bindings/classical_ols.mojo) calls it through the "
                            "connected decomp entries for an unweighted LinearRegression.fit with an "
                            "intercept and at most 511 columns"),
    "ols_fit_resident": ("n/a", [], _FAST_LINEAR),
    "ridge_fit_resident": ("n/a", [], _FAST_LINEAR),
}

#: (entry, binding source) -> (status, lanes, why): a row whose answer
#: differs between a GPU binding and its host twin.
ANNOTATIONS_BY_SRC = {
    **{(entry, "bindings/_mojolearn_preprocessing_host.mojo"): (
        "covered", [lane],
        "Source reach: MOJOLEARN_VENDOR=cpu maps the ordinary preprocessing binding to "
        "preprocessing_host; the native scaler owner calls its fixed " + entry +
        " entry (finite, unweighted input for standard_fit). Existing lane references "
        "are retained separately; this annotation is not execution of the new native door.")
       for entry, lane in (("standard_fit", "standard-scaler"),
                           ("standard_transform", "standard-scaler"),
                           ("minmax_fit", "minmax-scaler"),
                           ("minmax_transform", "minmax-scaler"))},
    **{(entry, "bindings/_mojolearn_estimators_host.mojo"): (
        "owed", [lane],
        "Source reach requires MOJOLEARN_IDENTITY_HOST_INFER=" + lane +
        ": the lane saves its scaler, host_model loads Host" + estimator +
        ", and the held-out transform uses the estimators_host entry. Ordinary "
        "CPU records exercise preprocessing_host instead. Bounded local CPU host-infer "
        "execution passed at e7c23510b (base fixture, one repeat; retained "
        "scaler-host-infer-cpu-e7c23510b.result.json). Matching installed-package and "
        "reference qualification are still owed; this is not full fixture coverage.")
       for entry, lane, estimator in (("standard_transform", "standard-scaler", "StandardScaler"),
                                     ("minmax_transform", "minmax-scaler", "MinMaxScaler"))},
    **{(e, "bindings/_mojolearn_training_host.mojo"): ("n/a", [], _HOST_SLOT)
       for e in ("optimizer_resident_open", "optimizer_resident_download",
                 "optimizer_resident_upload")},
    **{(e, "bindings/_mojolearn_training.mojo"): (
        "covered", ["samba-optimizer-state", "samba"],
        _SAMBA_OWNER + ": _samba_open_optimizer opens the slot (optimizer_resident_open), the snapshot "
        "downloads it (state_dict), the publish uploads into it (load_state_dict) and finalization closes "
        "it; the optimizer door's own slots move through its kernels instead")
       for e in ("optimizer_resident_open", "optimizer_resident_close", "optimizer_resident_download",
                 "optimizer_resident_upload")},
}

# Runtime-owner migration (2026-10-22): these are reviewed source routes,
# not execution, reference or fault-injection admission. Fixed import-time
# symbol tables do not prove that a registered ABI wrapper is ever called.
ANNOTATIONS_BY_SRC.update(
    {('seq_optimizer_init', 'bindings/_mojolearn_x_sequence.mojo'): ('covered',
                                                                     ['sequence-rmsprop'],
                                                                     'RMSprop.__init__ -> '
                                                                     '_SEQ_NATIVE(seq_construct) -> '
                                                                     'seq_construct -> verified '
                                                                     'selected[seq_optimizer_init]; '
                                                                     'source reach only, qualification '
                                                                     'remains recorded separately'),
     ('seq_optimizer_step', 'bindings/_mojolearn_x_sequence.mojo'): ('covered',
                                                                     ['sequence-rmsprop'],
                                                                     'RMSprop.step -> '
                                                                     '_SEQ_NATIVE(seq_step) -> seq_step '
                                                                     '-> verified '
                                                                     'entries[seq_optimizer_step]; '
                                                                     'source reach only, qualification '
                                                                     'remains recorded separately'),
     ('seq_optimizer_state', 'bindings/_mojolearn_x_sequence.mojo'): ('covered',
                                                                      ['sequence-rmsprop'],
                                                                      'opt.state in _sequence_opt_run -> '
                                                                      '_SEQ_NATIVE(seq_state) -> '
                                                                      'verified '
                                                                      'entries[seq_optimizer_state]; '
                                                                      'source reach only, qualification '
                                                                      'remains recorded separately'),
     ('seq_optimizer_close', 'bindings/_mojolearn_x_sequence.mojo'): ('covered',
                                                                      ['sequence-rmsprop'],
                                                                      '_sequence_opt_run optimizer '
                                                                      'finalization -> '
                                                                      '_SEQ_NATIVE(seq_finalize) -> '
                                                                      'verified '
                                                                      'entries[seq_optimizer_close]; '
                                                                      'source reach only, qualification '
                                                                      'remains recorded separately'),
     ('seq_optimizer_export', 'bindings/_mojolearn_x_sequence.mojo'): ('covered',
                                                                       ['sequence-optimizer-state'],
                                                                       'NAdam/RMSprop.state_dict -> '
                                                                       '_SEQ_NATIVE(seq_export) -> '
                                                                       'verified '
                                                                       'entries[seq_optimizer_export]; '
                                                                       'source reach only, qualification '
                                                                       'remains recorded separately'),
     ('seq_optimizer_import', 'bindings/_mojolearn_x_sequence.mojo'): ('covered',
                                                                       ['sequence-optimizer-state'],
                                                                       'NAdam/RMSprop.load_state_dict -> '
                                                                       '_SEQ_NATIVE(seq_import) -> '
                                                                       'verified '
                                                                       'entries[seq_optimizer_import]; '
                                                                       'source reach only, qualification '
                                                                       'remains recorded separately'),
     ('seq_adafactor_step', 'bindings/_mojolearn_x_sequence.mojo'): ('covered',
                                                                     ['sequence-adafactor'],
                                                                     'Adafactor.step -> '
                                                                     '_SEQ_NATIVE(seq_step), kind5 -> '
                                                                     'verified '
                                                                     'entries[seq_adafactor_step]; '
                                                                     'source reach only, qualification '
                                                                     'remains recorded separately'),
     ('seq_lamb_step', 'bindings/_mojolearn_x_sequence.mojo'): ('covered',
                                                                ['sequence-lamb'],
                                                                'LAMB.step -> _SEQ_NATIVE(seq_step), '
                                                                'kind6 -> verified '
                                                                'entries[seq_lamb_step]; source reach '
                                                                'only, qualification remains recorded '
                                                                'separately'),
     ('seq_optimizer_init', 'bindings/_mojolearn_x_sequence_host.mojo'): ('covered',
                                                                          ['sequence-rmsprop'],
                                                                          'RMSprop.__init__ -> '
                                                                          '_SEQ_NATIVE(seq_construct) -> '
                                                                          'seq_construct -> verified '
                                                                          'selected[seq_optimizer_init]; '
                                                                          'source reach only, '
                                                                          'qualification remains '
                                                                          'recorded separately'),
     ('seq_optimizer_step', 'bindings/_mojolearn_x_sequence_host.mojo'): ('covered',
                                                                          ['sequence-rmsprop'],
                                                                          'RMSprop.step -> '
                                                                          '_SEQ_NATIVE(seq_step) -> '
                                                                          'seq_step -> verified '
                                                                          'entries[seq_optimizer_step]; '
                                                                          'source reach only, '
                                                                          'qualification remains '
                                                                          'recorded separately'),
     ('seq_optimizer_state', 'bindings/_mojolearn_x_sequence_host.mojo'): ('covered',
                                                                           ['sequence-rmsprop'],
                                                                           'opt.state in '
                                                                           '_sequence_opt_run -> '
                                                                           '_SEQ_NATIVE(seq_state) -> '
                                                                           'verified '
                                                                           'entries[seq_optimizer_state]; '
                                                                           'source reach only, '
                                                                           'qualification remains '
                                                                           'recorded separately'),
     ('seq_optimizer_close', 'bindings/_mojolearn_x_sequence_host.mojo'): ('covered',
                                                                           ['sequence-rmsprop'],
                                                                           '_sequence_opt_run optimizer '
                                                                           'finalization -> '
                                                                           '_SEQ_NATIVE(seq_finalize) -> '
                                                                           'verified '
                                                                           'entries[seq_optimizer_close]; '
                                                                           'source reach only, '
                                                                           'qualification remains '
                                                                           'recorded separately'),
     ('seq_optimizer_export', 'bindings/_mojolearn_x_sequence_host.mojo'): ('covered',
                                                                            ['sequence-optimizer-state'],
                                                                            'NAdam/RMSprop.state_dict -> '
                                                                            '_SEQ_NATIVE(seq_export) -> '
                                                                            'verified '
                                                                            'entries[seq_optimizer_export]; '
                                                                            'source reach only, '
                                                                            'qualification remains '
                                                                            'recorded separately'),
     ('seq_optimizer_import', 'bindings/_mojolearn_x_sequence_host.mojo'): ('covered',
                                                                            ['sequence-optimizer-state'],
                                                                            'NAdam/RMSprop.load_state_dict '
                                                                            '-> _SEQ_NATIVE(seq_import) '
                                                                            '-> verified '
                                                                            'entries[seq_optimizer_import]; '
                                                                            'source reach only, '
                                                                            'qualification remains '
                                                                            'recorded separately'),
     ('seq_adafactor_step', 'bindings/_mojolearn_x_sequence_host.mojo'): ('covered',
                                                                          ['sequence-adafactor'],
                                                                          'Adafactor.step -> '
                                                                          '_SEQ_NATIVE(seq_step), kind5 '
                                                                          '-> verified '
                                                                          'entries[seq_adafactor_step]; '
                                                                          'source reach only, '
                                                                          'qualification remains '
                                                                          'recorded separately'),
     ('seq_lamb_step', 'bindings/_mojolearn_x_sequence_host.mojo'): ('covered',
                                                                     ['sequence-lamb'],
                                                                     'LAMB.step -> '
                                                                     '_SEQ_NATIVE(seq_step), kind6 -> '
                                                                     'verified entries[seq_lamb_step]; '
                                                                     'source reach only, qualification '
                                                                     'remains recorded separately'),
     ('optimizer_resident_open', 'bindings/_mojolearn_x_sequence.mojo'): ('n/a',
                                                                          [],
                                                                          'Import-time native '
                                                                          'registration/admission probes '
                                                                          'retain the symbol; public '
                                                                          'operations call seq_* '
                                                                          'entries. '
                                                                          'sequence.opt_public_device '
                                                                          'directly invokes the typed '
                                                                          'resident kernel, bypassing '
                                                                          'this wrapper. The typed '
                                                                          'kernel opt_resident_open is '
                                                                          'reached through the seq_* '
                                                                          'owner. Typed-kernel source '
                                                                          'lane: sequence-rmsprop.'),
     ('optimizer_resident_close', 'bindings/_mojolearn_x_sequence.mojo'): ('n/a',
                                                                           [],
                                                                           'Import-time native '
                                                                           'registration/admission '
                                                                           'probes retain the symbol; '
                                                                           'public operations call seq_* '
                                                                           'entries. '
                                                                           'sequence.opt_public_device '
                                                                           'directly invokes the typed '
                                                                           'resident kernel, bypassing '
                                                                           'this wrapper. The typed '
                                                                           'kernel opt_resident_close is '
                                                                           'reached through the seq_* '
                                                                           'owner. Typed-kernel source '
                                                                           'lane: sequence-rmsprop.'),
     ('optimizer_resident_move', 'bindings/_mojolearn_x_sequence.mojo'): ('n/a',
                                                                          [],
                                                                          'Import-time native '
                                                                          'registration/admission probes '
                                                                          'retain the symbol; public '
                                                                          'operations call seq_* '
                                                                          'entries. '
                                                                          'sequence.opt_public_device '
                                                                          'directly invokes the typed '
                                                                          'resident kernel, bypassing '
                                                                          'this wrapper. The typed '
                                                                          'kernel opt_resident_move is '
                                                                          'reached through the seq_* '
                                                                          'owner. Typed-kernel source '
                                                                          'lane: sequence-rmsprop.'),
     ('optimizer_resident_step', 'bindings/_mojolearn_x_sequence.mojo'): ('n/a',
                                                                          [],
                                                                          'Import-time native '
                                                                          'registration/admission probes '
                                                                          'retain the symbol; public '
                                                                          'operations call seq_* '
                                                                          'entries. '
                                                                          'sequence.opt_public_device '
                                                                          'directly invokes the typed '
                                                                          'resident kernel, bypassing '
                                                                          'this wrapper. The typed '
                                                                          'kernel opt_resident_step is '
                                                                          'reached through the seq_* '
                                                                          'owner. Typed-kernel source '
                                                                          'lane: sequence-rmsprop.'),
     ('optimizer_resident_step_dev', 'bindings/_mojolearn_x_sequence.mojo'): ('n/a',
                                                                              [],
                                                                              'Import-time native '
                                                                              'registration/admission '
                                                                              'probes retain the symbol; '
                                                                              'public operations call '
                                                                              'seq_* entries. '
                                                                              'sequence.opt_public_device '
                                                                              'directly invokes the '
                                                                              'typed resident kernel, '
                                                                              'bypassing this wrapper. '
                                                                              'The typed kernel '
                                                                              'opt_resident_step_dev is '
                                                                              'reached through the seq_* '
                                                                              'owner. Typed-kernel '
                                                                              'source lane: '
                                                                              'sequence-device-tensor.'),
     ('adafactor_resident_open', 'bindings/_mojolearn_x_sequence.mojo'): ('n/a',
                                                                          [],
                                                                          'Import-time native '
                                                                          'registration/admission probes '
                                                                          'retain the symbol; public '
                                                                          'operations call seq_* '
                                                                          'entries. '
                                                                          'sequence.opt_public_device '
                                                                          'directly invokes the typed '
                                                                          'resident kernel, bypassing '
                                                                          'this wrapper. The typed '
                                                                          'kernel '
                                                                          'adafactor_resident_open is '
                                                                          'reached through the seq_* '
                                                                          'owner. Typed-kernel source '
                                                                          'lane: sequence-adafactor.'),
     ('adafactor_resident_step', 'bindings/_mojolearn_x_sequence.mojo'): ('n/a',
                                                                          [],
                                                                          'Import-time native '
                                                                          'registration/admission probes '
                                                                          'retain the symbol; public '
                                                                          'operations call seq_* '
                                                                          'entries. '
                                                                          'sequence.opt_public_device '
                                                                          'directly invokes the typed '
                                                                          'resident kernel, bypassing '
                                                                          'this wrapper. The typed '
                                                                          'kernel '
                                                                          'adafactor_resident_step is '
                                                                          'reached through the seq_* '
                                                                          'owner. Typed-kernel source '
                                                                          'lane: sequence-adafactor.'),
     ('adafactor_resident_step_dev', 'bindings/_mojolearn_x_sequence.mojo'): ('n/a',
                                                                              [],
                                                                              'Import-time native '
                                                                              'registration/admission '
                                                                              'probes retain the symbol; '
                                                                              'public operations call '
                                                                              'seq_* entries. '
                                                                              'sequence.opt_public_device '
                                                                              'directly invokes the '
                                                                              'typed resident kernel, '
                                                                              'bypassing this wrapper. '
                                                                              'The typed kernel '
                                                                              'adafactor_resident_step_dev '
                                                                              'is reached through the '
                                                                              'seq_* owner. Its '
                                                                              'device-specific identity '
                                                                              'lane is still owed; no '
                                                                              'reference is claimed.'),
     ('lamb_resident_open', 'bindings/_mojolearn_x_sequence.mojo'): ('n/a',
                                                                     [],
                                                                     'Import-time native '
                                                                     'registration/admission probes '
                                                                     'retain the symbol; public '
                                                                     'operations call seq_* entries. '
                                                                     'sequence.opt_public_device '
                                                                     'directly invokes the typed '
                                                                     'resident kernel, bypassing this '
                                                                     'wrapper. The typed kernel '
                                                                     'lamb_resident_open is reached '
                                                                     'through the seq_* owner. '
                                                                     'Typed-kernel source lane: '
                                                                     'sequence-lamb.'),
     ('lamb_resident_step', 'bindings/_mojolearn_x_sequence.mojo'): ('n/a',
                                                                     [],
                                                                     'Import-time native '
                                                                     'registration/admission probes '
                                                                     'retain the symbol; public '
                                                                     'operations call seq_* entries. '
                                                                     'sequence.opt_public_device '
                                                                     'directly invokes the typed '
                                                                     'resident kernel, bypassing this '
                                                                     'wrapper. The typed kernel '
                                                                     'lamb_resident_step is reached '
                                                                     'through the seq_* owner. '
                                                                     'Typed-kernel source lane: '
                                                                     'sequence-lamb.'),
     ('lamb_resident_step_dev', 'bindings/_mojolearn_x_sequence.mojo'): ('n/a',
                                                                         [],
                                                                         'Import-time native '
                                                                         'registration/admission probes '
                                                                         'retain the symbol; public '
                                                                         'operations call seq_* entries. '
                                                                         'sequence.opt_public_device '
                                                                         'directly invokes the typed '
                                                                         'resident kernel, bypassing '
                                                                         'this wrapper. The typed kernel '
                                                                         'lamb_resident_step_dev is '
                                                                         'reached through the seq_* '
                                                                         'owner. Its device-specific '
                                                                         'identity lane is still owed; '
                                                                         'no reference is claimed.'),
     ('optimizer_resident_step', 'bindings/_mojolearn_training.mojo'): ('covered',
                                                                        ['samba-untied-dropout-accum'],
                                                                        'SambaStack.train_step -> '
                                                                        'samba_train_step; A4 skips '
                                                                        'dedicated Samba resident '
                                                                        'shortcut, dropout.1 also '
                                                                        'refuses shortcut; '
                                                                        'maximize=False and host '
                                                                        'parameter branch calls verified '
                                                                        'optimizer_resident_step; source '
                                                                        'reach only, qualification '
                                                                        'remains recorded separately'),
     ('optimizer_resident_close', 'bindings/_mojolearn_training_host.mojo'): ('covered',
                                                                              ['optim-sgd'],
                                                                              'public_step_host opens '
                                                                              'host slot for default '
                                                                              'resident=None -> '
                                                                              'keep_slot retains '
                                                                              'training_host module -> '
                                                                              'optimizer __del__ calls '
                                                                              'optimizer_finalize_binding '
                                                                              '-> compiled resident '
                                                                              'owner close entry; source '
                                                                              'reach only, qualification '
                                                                              'remains recorded '
                                                                              'separately')}
)


# Native-owner tables retain their compiled callees at import, so the Python
# graph cannot see these specific calls. Record only the lane's actual public
# operation; a step does not exercise state setters, zeroing or checkpoint IO.
ANNOTATIONS_BY_SRC.update({
    **{("lr_schedule_at", src): (
        "covered", ["sequence-lr-schedulers"],
        "Source reach: sequence-lr-schedulers hashes StepLR/ExponentialLR/OneCycleLR.bits_at; "
        "_OPT_NATIVE(schedule_bits) -> schedule_operation -> retained lr_schedule_at. "
        "_lr_native uses the identical base binding (core_host for the CPU package). "
        "Source accounting only; reference and execution admission stay separate.")
       for src in ("bindings/_mojolearn.mojo", "bindings/_mojolearn_core_host.mojo")},
    **{("lr_schedule_series", src): (
        "owed", ["native-schedule-series"],
        "New source route: schedule lr_values/lrs -> native schedule_values/schedule_list -> "
        "retained lr_schedule_series. The new lane consumes both outputs over phase boundaries "
        "and empty series; coverage-native-schedule-series faults returned values. Bounded CPU/Metal clean "
        "runs passed on native43 with harness7f64. Fault/restoration records, all-fixture cross-vendor "
        "references and installed-package admission remain pending.")
       for src in ("bindings/_mojolearn.mojo", "bindings/_mojolearn_core_host.mojo")},
    **{(entry, src): (
        "covered", [lane],
        "Source reach: " + operation + " -> _OPT_NATIVE(" + route + ") -> optimizer_native "
        "-> retained " + entry + "; selected training binding maps to training_host on CPU. "
        "The named lane consumes the changed output. This is not new execution or fault evidence.")
       for src in ("bindings/_mojolearn_training.mojo", "bindings/_mojolearn_training_host.mojo")
       for entry, lane, operation, route in (
           ("optimizer_door_step", "optim-sgd", "SGD.step", "step"),
           ("optimizer_door_state", "optim-maximize", "SGD.exp_avg", "state"),
           ("optimizer_door_clip", "optim-adam-clip", "training.clip_grad_norm_", "clip"),
           ("optimizer_door_accumulate", "grad-accumulation", "training.accumulate_grads", "accumulate"))},
    **{(entry, src): (
        "owed", lanes,
        "New source route: native public optimizer " + route + " -> retained " + entry + ". "
        "The named new lanes consume the actual bare owner operation. CPU IO scope is query "
        "and refusal only; optim-resident-io qualifies GPU transfers. Independent new sabotage "
        "arms are authored. Bounded CPU/Metal clean runs passed on native43 with harness7f64; "
        "fault/restoration, all-fixture reference and installed-package admission remain pending.")
       for src in ("bindings/_mojolearn_training.mojo", "bindings/_mojolearn_training_host.mojo")
       for entry, route, lanes in (
           ("optimizer_door_io", "io/raw_io/param_device",
            ["optim-io-capability"] + (["optim-resident-io"] if src.endswith("_training.mojo") else [])),
           ("optimizer_door_state_set", "state_set", ["optim-owner-lifecycle"]),
           ("optimizer_door_load_state", "load", ["optim-owner-lifecycle"]),
           ("optimizer_door_zero", "zero", ["optim-owner-lifecycle"]))},
    **{(entry, "bindings/_mojolearn_x_sequence.mojo"): (
        "covered", ["sequence-device-tensor"],
        "Source reach on a GPU column: sequence-device-tensor consumes to_device, tensor.copy "
        "and tensor download through _OPT_NATIVE(seqdev_*) -> sequence_device_operation -> "
        "retained " + entry + ". CPU's host-array route does not qualify this device entry. "
        "Source accounting only; retained execution/reference/fault scopes remain separate.")
       for entry in ("seq_tensor_upload", "seq_tensor_download", "seq_tensor_copy")},
    ("seq_tensor_free", "bindings/_mojolearn_x_sequence.mojo"): (
        "owed", ["sequence-device-release"],
        "New GPU-only source route: native sequence finalization and failed upload/copy cleanup "
        "-> retained seq_tensor_free. The new lane independently checks pool closure after explicit, "
        "repeated, GC and forced native-operation failures; a no-op free fault is authored. CPU has "
        "no resident pool. Bounded Metal clean execution passed on native43 with harness7f64. "
        "Fault/restoration, all-fixture reference and installed-package admission remain pending."),
})


# lane verify-gaps-20261022: the rows the 2026-10-22 audit read GAP. The
# native owners below retain these entries at import (a context table, a
# class slot or a finalizer descriptor), so the Python graph cannot see the
# call; each annotation names the public operation the lane runs and the
# native call site that takes the entry.
_HOST_TREE = ("the saved-model owner (bindings/host_tree_native.mojo) calls the forest binding's entry by "
              "name from the table _native_host_archive_bridge.connect hands it at import "
              "(`host_tree_entries`); ")
_FOREST = "bindings/_mojolearn_forest_host.mojo"
_TOK = "bindings/_mojolearn_tokenizer_host.mojo"
ANNOTATIONS_BY_SRC.update({
    ("forest_host_rf_predict_proba", _FOREST): (
        "covered", ["saved-model-host-infer"],
        _HOST_TREE + "HostForest.predict / predict_proba on the lane's saved RandomForestClassifier "
        "(sequential engine) take it in host_tree_vote"),
    ("forest_host_groves_predict", _FOREST): (
        "covered", ["saved-model-host-infer"],
        _HOST_TREE + "HostForest.predict on the lane's parallel_groves RandomForestRegressor takes it in "
        "host_tree_vote"),
    ("forest_host_groves_release", _FOREST): (
        "covered", ["saved-model-host-infer"],
        _HOST_TREE + "the groves HostForest's finalizer (host_archive_finalize_call, installed as the "
        "class's __del__ at import) releases its prepared groves through it when the lane drops the model"),
    ("forest_host_gbdt_predict", _FOREST): (
        "covered", ["saved-model-host-infer"],
        _HOST_TREE + "HostGBDT.predict / predict_proba on the lane's saved GradientBoosting take it in "
        "host_gbdt_predict"),
    ("forest_host_gbdt_sigmoid_pair", _FOREST): (
        "covered", ["saved-model-host-infer"],
        _HOST_TREE + "HostGBDT.predict_proba on the lane's saved Logloss GradientBoosting takes it"),
    ("forest_host_et_predict", _FOREST): (
        "covered (new lane)", ["saved-model-host-infer-trees"],
        _HOST_TREE + "HostForest.predict / predict_proba on the lane's saved ExtraTreesClassifier and "
        "ExtraTreesRegressor take it in host_tree_vote"),
    ("forest_host_rf_predict_reg", _FOREST): (
        "covered (new lane)", ["saved-model-host-infer-trees"],
        _HOST_TREE + "HostForest.predict and host_predict on the lane's saved RandomForestRegressor "
        "(default engine) take it in host_tree_vote"),
    ("forest_host_gbdt_expand_ctr", _FOREST): (
        "covered", ["gbdt-categorical-ctr-tables", "gbdt-tensor-ctr-tables"],
        _HOST_TREE + "on a CPU column the lanes predict through HostGBDT of the GPU-saved CTR model "
        "(identity_break _ctr_saved_or_fit), whose CTR tables make host_gbdt_predict expand the input "
        "through it"),
    ("host_archive_finalize_call", _TOK): (
        "covered", ["saved-model-host-infer"],
        "_native_host_archive_bridge.connect installs it, through classical_attribute_descriptor, as the "
        "__del__ of every saved host model class with a finalizer and of HostForest; the lane's groves "
        "HostForest releases its groves through it (forest_host_groves_release)"),
    ("classical_attribute_call", _TOK): (
        "covered", ["ols", "ridge", "ols-no-intercept", "ridge-no-intercept", "ols-weighted"],
        "_native_classical_bridge.connect_public installs it, through classical_attribute_descriptor, as "
        "LinearRegression's and Ridge's __getattribute__ at import; every coef_, intercept_ and predict "
        "the lanes read goes through it"),
    ("cross_vendor_call", _TOK): (
        "covered (new lane)", ["cross-vendor-fold"],
        "mojolearn.cross_vendor binds it at import (`_native_cross_vendor`); ordered_fold and fold_pair "
        "call it"),
    ("native_process_call", _TOK): (
        "covered", ["par-mlp", "par-samba", "par-samba-clip"],
        "mojolearn._parallel_neural imports _native_process, whose import-time `initialize` call sets the "
        "interpreter and environment the native process pool (bindings/native_process_pool.mojo) spawns "
        "the parallel trainers' workers with; par-* lanes run on two-device boxes"),
    ("parallel_mlp_call", _TOK): (
        "covered", ["par-mlp"],
        "mojolearn._parallel_neural imports _parallel_mlp, whose import-time `initialize` call hands the "
        "parallel MLP owner (bindings/parallel_mlp_native.mojo) SmallMLPTrainer, AdamW and the training "
        "binding it builds its replicas from; par-* lanes run on two-device boxes"),
    ("pca_native_call", _TOK): (
        "covered", ["pca", "pca-whiten", "pca-inverse", "pca-whiten-inverse", "pca-full-whiten",
                    "x-decomp-pca-randomized", "classical-float16-sequences"],
        "_native_pca_bridge.connect hands it to the public PCA class at import; PCA.fit, transform and "
        "inverse_transform call it"),
    ("arena_rows_native_call", _TOK): (
        "covered", ["cross-val", "x-metrics-search", "x-metrics-classification"],
        "model_selection's import connects it (_native_arena_rows.connect sets _arena_io._ROWS_CALL); "
        "DeviceCache.supports, id_of, get and DeviceRows call it on the cross-validation and metrics paths"),
    ("ce_loss_dev", "bindings/_mojolearn_training.mojo"): (
        "covered (new lane)", ["training-device-cross-entropy"],
        "_native_optimizer_bridge retains it at import with the training binding's entries; "
        "training.cross_entropy on a TrainDeviceArray of logits takes it "
        "(bindings/training_cross_entropy_native.mojo, `ce_loss_dev`); GPU columns only"),
})


#: (public, op) -> (status, lanes, why) for the public methods.
# RobustScaler's native owner is new; these existing lanes consume its public
# fit/transform outputs but do not supply current-freeze qualification yet.
ANNOTATIONS_BY_SRC[("robust_scaler_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", ["x-prep-robust-scaler", "x-prep-robust-scaler-unit-variance"],
    "Public RobustScaler lifecycle calls robust_lifecycle_c, which prepares native buffers and "
    "runs the existing x_prep quantile/scale/affine stages. Existing lanes consume center_, scale_ "
    "and transformed outputs; new public lifecycle/refusal tests are authored. New source only: "
    "current runtime execution, fault/restoration, installed-wheel and matching reference admission remain pending.")


# The stateless x_prep owner is new; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("prep_stateless_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", ["x-prep-binarizer", "x-prep-normalizer", "x-prep-polynomial-features",
             "x-prep-variance-threshold", "x-prep-select-kbest", "x-prep-rfe",
             "x-prep-quantile-transformer", "x-prep-power-transformer", "x-prep-inverse-transforms"],
    "Public Binarizer, Normalizer, PolynomialFeatures, VarianceThreshold, QuantileTransformer and PowerTransformer "
    "fit/transform/fit_transform/inverse_transform "
    "and the selectors' transform call prep_stateless_c, which builds the existing binarize/normalize/"
    "poly/p2m_transpose, col_stats/var_ptp/c2_key64/c2_gt and gather_cols x_prep programs natively. New source only: runtime execution and identity admission remain pending.")


# QNRegressor's native owner is new; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("qn_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    "Public QNRegressor fit/predict/score call qn_c, which runs the existing qn_fit and "
    "qn_decision_function entries and the x_metrics R^2 program natively. New source only: "
    "runtime execution and identity admission remain pending.")


# A new native owner door; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("prep_encode_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    'Public SimpleImputer, KBinsDiscretizer, OrdinalEncoder, OneHotEncoder, SplineTransformer fit/transform/inverse_transform/fit_transform and TargetEncoder.transform call prep_encode_c, which builds and runs their x_prep programs natively through xprep_run.' + " New source only: runtime execution and identity admission remain pending.")


# A new native owner door; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("prep_classify_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    'Public naive Bayes (Gaussian, Multinomial, Bernoulli, Complement, Categorical) and LDA/QDA fit, partial_fit, predict, predict_proba, predict_log_proba, predict_joint_log_proba, score, decision_function, transform and fit_transform call prep_classify_c, which builds their x_prep programs natively, runs them on the shared arena runner and scores through the native label and accuracy steps.' + " New source only: runtime execution and identity admission remain pending.")


# A new native owner door; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("tsvd_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    'Public TruncatedSVD fit/fit_transform/transform/inverse_transform call tsvd_c, which converts X and runs the existing tsvd entries through the compiled x_decomp_tsvd_randomized / x_decomp_tsvd_tsqr drivers natively.' + " New source only: runtime execution and identity admission remain pending.")


# A new native owner door; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("ipca_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    'Public IncrementalPCA fit/partial_fit/fit_transform/transform/inverse_transform call ipca_c, which converts X and runs one compiled x_decomp_ipca_fit / _partial / _project driver natively.' + " New source only: runtime execution and identity admission remain pending.")


# A new native owner door; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("jl_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    'Public johnson_lindenstrauss_min_dim calls jl_c, which checks eps and n_samples and computes the bound natively.' + " New source only: runtime execution and identity admission remain pending.")


# A new native owner door; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("dict_learning_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    'Public DictionaryLearning, MiniBatchDictionaryLearning, SparseCoder, SparsePCA, MiniBatchSparsePCA fit/fit_transform/transform/inverse_transform and sparse_encode call dict_learning_c, which runs the existing x_decomp entries through the native decomposition kit.' + " New source only: runtime execution and identity admission remain pending.")


# A new native owner door; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("pls_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    'Public PLSRegression, PLSCanonical and CCA fit/fit_transform/transform/inverse_transform/predict call pls_c, which runs the existing x_decomp entries through the native decomposition kit.' + " New source only: runtime execution and identity admission remain pending.")


# A new native owner door; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("mcd_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    'Public MinCovDet and EllipticEnvelope fit/score/mahalanobis/get_precision/score_samples/decision_function/predict/fit_predict call mcd_c, which runs the existing x_decomp entries through the native decomposition kit.' + " New source only: runtime execution and identity admission remain pending.")


# A new native owner door; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("als_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    'Public AlternatingLeastSquares fit/recommend/similar_items call als_c, which runs the existing x_decomp entries through the native decomposition kit.' + " New source only: runtime execution and identity admission remain pending.")


# A new native owner door; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("linalg_decomp_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    'Public randomized_svd, solve and lstsq call linalg_decomp_c, which runs the existing x_decomp entries through the native decomposition kit.' + " New source only: runtime execution and identity admission remain pending.")


# A new native owner door; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("decomp_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    "Public FastICA, NMF, FactorAnalysis, LatentDirichletAllocation, KernelPCA and Gaussian/SparseRandomProjection operations call decomp_c, which checks the inputs, runs one compiled x_decomp driver (plus KernelPCA's x_neighbors kernel steps) and stores the fitted attributes natively." + " New source only: runtime execution and identity admission remain pending.")


# LogisticRegression's native owner is new; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("logistic_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    "Public LogisticRegression fit/decision_function/predict/predict_proba/predict_log_proba/score "
    "call logistic_c, which runs the existing qn_fit, qn_decision_function, qn_predict_binary, "
    "qn_predict_multiclass, qn_sigmoid, qn_softmax and py2mojo_rows entries, the native label, "
    "weight and class-weight steps and the x_metrics accuracy program natively. New source only: "
    "runtime execution and identity admission remain pending.")


# The x_linear estimators' native owner is new; these lanes consume its public outputs.
ANNOTATIONS_BY_SRC[("x_linear_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    "Public x_linear estimators (SGD, GLMs, Huber, BayesianRidge/ARD, Lars/LassoLars, Quantile, "
    "Perceptron, PassiveAggressive, RidgeClassifier, SGDOneClassSVM, Isotonic, Lasso/ElasticNet) "
    "decision_function/predict/score/predict_proba and fits call x_linear_c, which runs the existing "
    "x_linear_fit / x_linear_decision entries, the native label and weight steps and the x_metrics "
    "programs natively. New source only: runtime execution and identity admission remain pending.")


# The native str-label encoder is new; these lanes consume its classes and codes.
ANNOTATIONS_BY_SRC[("str_labels_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", ["x-prep-label-encoder", "x-prep-label-binarizer", "x-prep-multilabel-binarizer"],
    "_labels.encode_labels / unique_inverse and LabelEncoder, LabelBinarizer and "
    "MultiLabelBinarizer transform hand str labels to str_labels_c (code-point order, "
    "first-seen classes). New source only: runtime execution remains pending.")


# Static lane reachability is not an execution/installed/fault receipt for a
# newly registered public attribute or automatic device finalizer entry.
for _new_entry in ("host_svr_attribute_call", "host_svc_attribute_call", "optimizer_device_finalize"):
    ANNOTATIONS_BY_SRC[(_new_entry, "bindings/_mojolearn_tokenizer_host.mojo")] = (
        "owed", [],
        "Targeted public operation and ownership tests are registered; retain their exact frozen "
        "execution scope separately. Matching installed-package, reference and fault/restoration "
        "admission for this entry remains pending; static lane reachability does not establish it.")


# The public linear/RMS owners retain these existing numerical entries at import.
# Source reach is distinct from execution of the new native preparation route.
for _src in ("bindings/_mojolearn_training.mojo", "bindings/_mojolearn_training_host.mojo"):
    for _entry in ("linear_forward", "linear_backward", "rms_norm_forward", "rms_norm_backward"):
        _owner = "training_rms_operation" if _entry.startswith("rms_") else "training_linear_operation"
        ANNOTATIONS_BY_SRC[(_entry, _src)] = (
            "covered", ["training-primitives"],
            "Source reach: public training." + _entry + " -> optimizer_native -> "
            + _owner + " -> retained numerical entry. The existing training-primitives "
            "lane consumes these results; current native preparation, installed-package and "
            "matching reference/fault qualification remain separately pending.")
for _entry in ("linear_forward_dev", "linear_backward_dev"):
    ANNOTATIONS_BY_SRC[(_entry, "bindings/_mojolearn_training.mojo")] = (
        "owed", [],
        "Public native linear owner routes all-device inputs to this existing GPU entry. "
        "Selected consumed-output and rollback tests are authored; a matching verifier lane, "
        "reference/fault/restoration and installed-package admission remain pending.")


# The user-object reader is new (lane purge-user-objects): it reads label lists,
# index pairs, group ids and nested lists for the classifiers, splitters, the
# CatBoost-style pool, UMAP's categorical target and the label metrics.
ANNOTATIONS_BY_SRC[("user_objects_native_call", "bindings/_mojolearn_tokenizer_host.mojo")] = (
    "owed", [],
    "model_selection (take, first_seen, kinds, scatter, column_f32), ensemble (group_runs, pairs, "
    "kinds), resample, linear_model, _metrics_impl, _umap_impl (sorted_codes), _array (nested), "
    "_expansion_trees and _expansion_prep hand Python objects to user_objects_c. New source only: "
    "runtime execution and a verifier lane remain pending.")


# lane verify-coverage-20261024: the whole-surface audit's last GAP rows,
# each checked in the source.
_ARENA_ROWS = ("_native_arena_rows connects msel_put, msel_free and msel_take_rows together as the "
               "owner context's arena_msel_put / _free / _take_rows; the arena owners "
               "(bindings/arena_resident_native.mojo, arena_materialize_native.mojo) put, gather and free "
               "every model_selection._Resident snapshot with them, which GridSearchCV and cross_validate "
               "take per fold (x-metrics-search reaches msel_free by name)")
ANNOTATIONS.update({
    **{e: ("covered", ["x-metrics-search"], _ARENA_ROWS) for e in ("msel_put", "msel_take_rows")},
    "nsum_f64": ("n/a", [], "a test seam: test_py_bugs.py holds the live Neumaier fold to CPython 3.12's sum "
                            "through _portable_math.nsum and this export only"),
    "gemm_fused": ("n/a", [], "fast tier only (Apple): linalg_fast.matmul_fused, built only with "
                              "MOJOLEARN_AFN_GEMM_EPILOGUE; the verifier runs identical and refuses a fast process"),
    "host_archive_finalize_call": ("n/a", [], "a lifecycle finalizer: the saved-host-model archive's cleanup "
                                              "descriptor frees its native buffers when the archive is collected; "
                                              "it returns nothing a digest can hold"),
    "causal_lm_blocks": ("n/a", [], "an accessor: CausalLM.blocks returns list(model._blocks), no arithmetic; the "
                                    "blocks it returns are the ones every hf-causal-lm forward runs"),
    "transformer_session_discard_tape": ("covered", ["transformer-forward-tape"],
                                         "TransformerForwardTape.close; the lane closes a tape, requires the "
                                         "backward to refuse it and the next tape's gradients to be unchanged"),
    "var_forecast": ("covered", ["sequence-var"],
                     "VAR.fit's owner (bindings/xseq_forecast_native.mojo) builds the VARResults the bridge "
                     "registered (_native_xseq_bridge.register(xseq_var_results=...)); its forecast calls "
                     "var_forecast, and sequence-var digests a.forecast(y, 10)"),
})

# lane coverage-gap-lanes-20261025: the 0.8.57 changed-scope audit's GAP rows.
# Each entry is installed or handed over at import, so the static reach cannot
# see the call; each was followed in the source to a recorded lane or to the
# new lane of tools/identity_lanes/public_native_params.py.
_NATIVE_PARAMS = ("public-set-params (tools/identity_lanes/public_native_params.py) constructs "
                  "HuberRegressor, PolynomialFeatures and GaussianRandomProjection, sets their parameters "
                  "with set_params, refits and digests the outputs and the parameters read back")
ANNOTATIONS.update({
    "msel_rebase_offsets_i32": (
        "covered", ["par-forest", "par-forest-et", "par-forest-et-clf", "par-forest-reg"],
        "parallel_ensemble.fit_forest is one native call (bindings/parallel_shard_native.mojo "
        "fit_forest -> _fit_forest), which rebases the tree ranges' node offsets through "
        "_rebase_offsets with the worker binding's msel_rebase_offsets_i32 "
        "(_native_parallel_shard._WORKER_BINDINGS['_mojolearn'], the core host twin on a CPU install); "
        "the four lanes fit 16 trees in four ranges and are held to the plain fit"),
    "shard_topk_merge_f32": (
        "covered", ["par-reference-knn", "par-reference-knn-reg"],
        "ReferenceShardedNeighbors.kneighbors/predict/predict_proba are one native call "
        "(bindings/parallel_shard_native.mojo reference_query -> _reference_query), which merges every "
        "reference block's candidates with the reference_merge entry _native_parallel_shard._connect_merge "
        "resolved (_buffer._native('shard_topk_merge_f32')); both lanes query four reference blocks and "
        "are held to the plain estimator"),
    "linear_constructor_native_call": (
        "covered", ["public-set-params", "x-huber-outliers"],
        "_native_linear_constructors.install replaces the __init__ of the 19 field-only linear models "
        "with a function calling this entry; x-huber-outliers constructs HuberRegressor(max_iter=30, "
        "epsilon=1.1) and digests its outliers_; " + _NATIVE_PARAMS),
    "full_reset_constructor_native_call": (
        "covered", ["kmeans", "public-set-params"],
        "_native_full_reset_params.install and _native_declared_constructors.install replace the __init__ of "
        "the 77 full-reset and 52 declared or plain field-only classes with a function calling this entry; the kmeans "
        "lane constructs KMeans(n_clusters=8, random_state=3) and digests its fit, and " + _NATIVE_PARAMS),
    "array_native_call": (
        "covered", ["kmeans", "public-set-params"],
        "every public mojolearn.Array operation (python/mojolearn/_array.py) is one call of this entry; "
        "identity_break._h digests each lane's outputs through numpy.asarray (the Array buffer/interface "
        "operations), and the kmeans lane's cluster_centers_/labels_ are Arrays"),
    "params_setter_native_call": (
        "covered", ["public-set-params"],
        "NativeParamsSetter.set_params (the linear models' set_params) calls this entry; " + _NATIVE_PARAMS),
    "declared_params_native_call": (
        "covered", ["public-set-params"],
        "_PrepBase.set_params and decomp _Base.set_params call this entry (_DECLARED_PARAMS_SET); "
        + _NATIVE_PARAMS),
    "public_attribute_native_call": (
        "covered", ["cholesky", "public-set-params"],
        "_native_public_attributes.connect_classes installs it as __getattribute__ of every connected "
        "class at import, so every attribute read of a fitted model runs it; the cholesky lane reads "
        "logdet_ (the cholesky_logdet property schema) and " + _NATIVE_PARAMS),
})

PY_ANNOTATIONS = {
    ("mojolearn.models.ParallelCausalLM", "generate"): (
        "covered", ["hf-causal-lm", "par-causal-lm"],
        "the inherited CausalLM.generate (hf-causal-lm) over ParallelCausalLM's forward (par-causal-lm)"),
}

#: Defines a verifier run provably turns DIVERGENT: the CPU identity gate builds
#: every host binding with it and requires the covered lanes' parts to move
#: (docs/VERIFY.md, tools/cpu_identity_gate_check.py).
GATE_DEFINES = frozenset({"MOJOLEARN_HOST_SABOTAGE"})


#: THE PRECISE-COVERAGE DECLARATIONS (lane precise-coverage-0859, 2026-10-26):
#: one decision per registration the static reach leaves "wide only" (a lane
#: runs the binding whole but no reach of its own names the export), each
#: followed in the source. TAB separated: entry, binding source (or `*` for
#: every registration of the entry), status, lanes, why.
#:   covered  the lanes (comma separated, each in the lane inventory) run a
#:            public call whose path reaches the export; `why` names that path
#:   n/a      a public path exists but no identical-mode lane can run it (the
#:            fast tier only, the verifier's own probes); `why` says which
#:   dead     no public path at all (no package code, native owner or lane
#:            calls it); `why` says what does name it. Listed for deletion.
DECLARED_TSV = os.path.join(ROOT, "tools", "identity_lanes", "coverage_declared.tsv")
DECLARED_STATUSES = ("covered", "n/a", "dead")
_DECLARED = None


def declared(path=None):
    """{(entry, src | "*"): (status, lanes, why)} from DECLARED_TSV."""
    global _DECLARED
    if _DECLARED is not None and path is None:
        return _DECLARED
    out = {}
    src_path = path or DECLARED_TSV
    try:
        fh = open(src_path, encoding="utf-8")
    except OSError:
        fh = None
    if fh is not None:
        with fh:
            for n, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line.strip() or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) != 5:
                    raise SystemExit(f"{src_path}:{n}: {len(cols)} columns, expected 5 (entry, src, status, lanes, why)")
                entry, src, status, lanes, why = cols
                lanes = [x for x in lanes.split(",") if x]
                if status not in DECLARED_STATUSES:
                    raise SystemExit(f"{src_path}:{n}: status {status!r}, expected one of {DECLARED_STATUSES}")
                if (status == "covered") != bool(lanes):
                    raise SystemExit(f"{src_path}:{n}: a covered row names its lanes; an n/a or dead row names none")
                if not why.strip():
                    raise SystemExit(f"{src_path}:{n}: no reason")
                if src != "*" and not src.startswith("bindings/"):
                    raise SystemExit(f"{src_path}:{n}: src {src!r} is a bindings/ path or *")
                if (entry, src) in out:
                    raise SystemExit(f"{src_path}:{n}: {entry} {src} declared twice")
                out[(entry, src)] = (status, lanes, why)
    if path is None:
        _DECLARED = out
    return out


def declaration(row):
    """The DECLARED_TSV row for one native row, or None."""
    table = declared()
    return table.get((row["entry"], row["src"])) or table.get((row["entry"], "*"))


DEAD_PREFIX = "dead (listed for deletion): "


def reach_state(row):
    """How one native row is reached, for the summary: `static` (a lane's own
    reach names it), `declared` (a hand-checked declaration or annotation names
    its lanes), `decided` (a declaration or annotation says n/a or dead, with
    its reason), `wide` (a lane runs the binding whole and nothing decided it)
    or `none`."""
    if row["lanes_precise"]:
        return "static"
    explicit = declaration(row) is not None or (row["entry"], row["src"]) in ANNOTATIONS_BY_SRC \
        or row["entry"] in ANNOTATIONS
    if explicit:
        status, lanes, _ = triage(row)
        if status.startswith("covered") and lanes:
            return "declared"
        if status == "n/a":
            return "decided"
    return "wide" if row["lanes_wide"] else "none"


def triage(row):
    """(status, lanes, why) for one native row."""
    key = (row["entry"], row["src"])
    decl = declaration(row)
    if decl is not None:
        status, lanes, why = decl
        if status == "covered":
            return "covered", sorted(set(lanes) | set(row["lanes_precise"])), why
        return "n/a", [], (DEAD_PREFIX + why) if status == "dead" else why
    if key in ANNOTATIONS_BY_SRC:
        status, lanes, why = ANNOTATIONS_BY_SRC[key]
        if status == "n/a":
            return status, [], why
        return status, sorted(set(lanes) | set(row["lanes_precise"])), why
    if row["entry"] in ANNOTATIONS:
        status, lanes, why = ANNOTATIONS[row["entry"]]
        return status, sorted(set(lanes) | set(row["lanes_precise"])), why
    if row["lanes_precise"]:
        return "covered", row["lanes_precise"], ""
    if row.get("constant_probe"):
        return "n/a", [], ("a constant probe: the body returns one compiled constant (a contract version, "
                           "vendor or capability flag), reads no data and computes nothing")
    if not row["py_files"]:
        if row.get("adapter_files"):
            return "n/a", [], ADAPTER_ONLY
        return "n/a", [], "no Python caller in the package (manifest, tests or Mojo-internal dispatch only)"
    return "GAP", [], "no lane reaches it"


#: Registrations whose whole job is a lifecycle step or a metadata answer:
#: a value fault has nothing to move, whatever helper their closure shares
#: (lane verify-coverage-20261024, each checked in the source). Only consulted
#: when no arm reaches the row.
_LIFECYCLE = "a lifecycle step (allocates, opens, frees or closes a resident object); no value leaves it"
_METADATA = "a metadata answer (a profile word, count or flag of the build or the object); no data value"
SABOTAGE_NA_ENTRIES = {
    **{e: _LIFECYCLE for e in (
        "kmeans_centers_release", "kernel_ridge_resident_release", "knn_index_release", "kde_fit_release",
        "forest_release_gpu", "moe_weights_free", "x_trees_dart_open", "x_trees_dart_close",
        "x_trees_tree_shap_cache_release", "causal_lm_session_create", "causal_lm_session_close",
        "msel_alloc", "msel_free", "x_prep_dev_free", "seq_tensor_alloc",
        "mlp_resident_open", "mlp_resident_close", "rf_data_session_open", "rf_data_session_close")},
    **{e: _METADATA for e in (
        "byte_lm_arithmetic_suffix", "byte_lm_host_arithmetic_suffix", "byte_lm_profile", "byte_lm_step_glue_arm",
        "cholesky_profile_jitter", "bpe_n_vocab", "bpe_max_token_bytes", "x_prep_dev_live",
        "gbdt_model_dim")},
}

# lane verify-coverage-20261024, after the arms' run on 9b04b92c5 (decisions in
# docs/plans/VERIFY_COVERAGE_20261024.md, "Arm decisions")
_RESAMPLE_OFF = ("compiled out of identical builds: RESAMPLE_GPU_GATHER is the fast tier or the opt-in "
                 "MOJOLEARN_CLASSICAL_C11_DRAW_GATHER, RESAMPLE_FAST_GATHER_NARROW the fast tier; in identical "
                 "the entry returns its not-taken flag (coverage-resample-gather MISSED on 9b04b92c5 for that reason)")
SABOTAGE_NA_ENTRIES.update({
    "resample_gather_gpu": _RESAMPLE_OFF, "resample_permutation_gather_gpu": _RESAMPLE_OFF,
    "resample_gather_narrow": _RESAMPLE_OFF,
    "xn_softmax64": "fast tier only: registered only under NC_PROBA64 (x_neighbors/proba64_nc.mojo, fast builds)",
    "nonzero_f32_count": ("the capacity count before the COO fill; a wrong count makes the fill refuse "
                          "(stale count), and the fill's values carry coverage-core-dense_coo"),
})

# lane coverage-sabotage-arms-20261025: the 0.8.57 coverage-source rows whose
# sabotage read `owed`, checked in the source at 4d7a0e8ad. Lane
# dead-native-entries-0859 (2026-10-26) deleted the nine with no live caller
# (adam_hyper_f64, lr_schedule_fill, gp_log64, gbdt_sigmoid,
# forest_host_gbdt_sigmoid, the device x_decomp_knn_select, byte_lm_launch_probe,
# x_neighbors_purity_flags, sched_exp_block; docs/TOMBSTONES.md). The two left
# are A/B tools' reach probes of FAST switches still in the tree.
_REACH_PROBE = "a quality/A-B tool's build or reach probe, no data value: "
SABOTAGE_NA_ENTRIES.update({
    "km_rbf_staged": _REACH_PROBE + ("1 when KM_FAST_RBF_STAGED is compiled in (tools/kfeat_quality.py, "
                                     "tools/kfeat_pair.py REACH_EXTRA), registered only under "
                                     "KM_FAST_RBF_RESIDENT (FAST + Apple)"),
    "x_decomp_mcd_cov_reach": _REACH_PROBE + ("MCD_ORDERED_COV's flag and reach counters "
                                              "(tools/mcd_ordered_quality.py)"),
})
#: (entry, binding) -> reason, for one registration of an entry its twin covers.
SABOTAGE_NA_BY_SRC = {
    ("nonzero_f32_fill", "bindings/_mojolearn.mojo"): (
        "the GPU base binding's COO scan: on a GPU install every spectral affinity takes the metrics dense "
        "entry, so no install reaches it through a public call (Metal unmoved under coverage-core-dense_coo, "
        "9b04b92c5); the host twin is armed"),
}


#: lane precise-coverage-0859 (2026-10-26): registrations made precise whose only
#: arms were for other lanes, and whose job is a lifecycle step, a flag or a
#: counter: no data value for a lane to move (each checked in the source).
SABOTAGE_NA_BY_SRC.update({
    ('forest_host_gbdt_ctr_sabotage', 'bindings/_mojolearn_forest_host.mojo'): (
        "metadata: returns the compile-time MOJOLEARN_GBDT_CTR_HOST_SABOTAGE flag (no data value); read only by the loader's sabotage refusal, and the define's own arm is what flips it."),
    ('forest_host_groves_sabotage', 'bindings/_mojolearn_forest_host.mojo'): (
        "metadata: returns the compile-time MOJOLEARN_FOREST_GROVES_SABOTAGE flag (no data value); read only by the loader's sabotage refusal, and the define's own arm is what flips it."),
    ('forest_host_sabotage', 'bindings/_mojolearn_forest_host.mojo'): (
        "metadata: returns the OR of the binary's three compile-time sabotage defines (no data value); read only by _backend's load-time sabotage refusal, and those defines' arms are what flip it."),
    ('mamba3_prefill_session_info', 'bindings/_mojolearn_mamba.mojo'): (
        "metadata: six counters (weight uploads/recopies/reuses, backward reuses/recomputes, stages_held) of the prefill session; no data value, the forward's bytes are covered by the mamba3 lane's own arms."),
    ('mamba3_session_close', 'bindings/_mojolearn_mamba.mojo'): (
        'lifecycle: closes the resident Mamba3DecodeSession (release of ctx, weights, state and stages); no data value.'),
    ('mamba3_session_create', 'bindings/_mojolearn_mamba.mojo'): (
        'lifecycle: allocates a closed Mamba3DecodeSession object; no GPU operation and no data value.'),
    ('causal_lm_blocks', 'bindings/_mojolearn_tokenizer_host.mojo'): (
        'Metadata: returns list(model._blocks), the per-layer config dicts; no computed data value. The lane only checks len(blocks) == n_layers, which fails closed by AssertionError, not a digest.'),
    ('native_process_call', 'bindings/_mojolearn_tokenizer_host.mojo'): (
        "Lifecycle: the lanes reach it only through the import-time 'initialize', which stores the interpreter path, environment mapping and vendor for the native process pool; no data value passes through it."),
    ('tokenizer_host_sabotage', 'bindings/_mojolearn_tokenizer_host.mojo'): (
        'Flag: reports whether the binding was built with a sabotage define; the loader refuses a flagged binding (a loader refusal never counts). No data value.'),
})


def sabotage_of(row):
    """(status, text) of one native row's fault-injection arm."""
    gate = sorted(set(row["sabotage_defines"]) & GATE_DEFINES)
    other = sorted(set(row["sabotage_defines"]) - GATE_DEFINES)
    patches = row["sabotage_patches"]
    bits = []
    if gate:
        bits.append("CPU gate " + gate[0].replace("MOJOLEARN_", ""))
    if other:
        bits.append(", ".join(d.replace("MOJOLEARN_", "") for d in other[:2]) + (" +%d" % (len(other) - 2)
                                                                               if len(other) > 2 else ""))
    cov = [x for x in patches if os.path.basename(x).startswith("coverage_")]
    seam = [x for x in patches if x not in cov]
    if cov:
        bits.append("coverage arm " + ", ".join(f"`{os.path.basename(x)[:-len('.patch')]}`" for x in cov[:2])
                    + (f" +{len(cov) - 2}" if len(cov) > 2 else ""))
    if seam:
        bits.append(f"{len(seam)} seam patch(es)")
    if (row["entry"], row.get("src")) in SABOTAGE_NA_BY_SRC:
        return "n/a", SABOTAGE_NA_BY_SRC[(row["entry"], row.get("src"))]
    if bits:
        return "armed", "; ".join(bits)
    if row["entry"] in SABOTAGE_NA_ENTRIES:
        return "n/a", SABOTAGE_NA_ENTRIES[row["entry"]]
    if not row.get("n_changed", 1) and "closure_kernel" in row:
        # an unchanged row (the whole-surface audit, --all): its closure decides
        if not row["closure_kernel"]:
            return "n/a", "door only: no kernel or impl definition in its closure (conversion, no arithmetic)"
        pending = [n for n in row["closure_kernel"] if _sab_na(n) is None]
        if not pending:
            return "n/a", _sab_na(row["closure_kernel"][0])
        return "owed", "no arm reaches its closure (" + pending[0] + (
            f" +{row['n_closure_kernel'] - 1}" if row["n_closure_kernel"] > 1 else "") + ")"
    if not row["changed_kernel"]:
        return "n/a", "door-only change (conversion, no arithmetic)"
    pending = [n for n in row["changed_kernel"] if _sab_na(n) is None]
    if not pending:
        return "n/a", _sab_na(row["changed_kernel"][0])
    return "owed", "no arm reaches " + pending[0]


_SAB_NA = None
_POINTER_ALIAS = re.compile(r"^_?[A-Z0-9]{1,4}P$")


def _sab_na(node):
    """The reason a changed definition carries no fault arm, or None
    (tools/identity_lanes/coverage_sabotage_na.tsv; pointer type aliases)."""
    global _SAB_NA
    if _SAB_NA is None:
        _SAB_NA = {}
        path = os.path.join(ROOT, "tools", "identity_lanes", "coverage_sabotage_na.tsv")
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip() and not line.startswith("#"):
                        key, _, why = line.rstrip("\n").partition("\t")
                        _SAB_NA[key] = why
        except OSError:
            pass
    f, _, name = node.partition(":")
    if _POINTER_ALIAS.match(name):
        return "a pointer type alias, no value"
    return _SAB_NA.get(node) or _SAB_NA.get(f + ":*")


def _short(lanes, n=3):
    if not lanes:
        return "-"
    return ", ".join(f"`{l}`" for l in lanes[:n]) + (f" +{len(lanes) - n}" if len(lanes) > n else "")


def audit_lanes(audit):
    """Include the complete lane inventory and explicit source annotations."""
    lanes = set(audit.get("lane_inventory", ()))
    for row in audit["native"]:
        lanes.update(triage(row)[1])
    for row in audit["python"]:
        lanes.update(row["lanes"])
        annotation = PY_ANNOTATIONS.get((row["public"], row["op"]))
        if annotation:
            lanes.update(annotation[1])
    return sorted(lanes)


def audit_references(audit):
    # Re-rendering an old audit must not substitute today's table evidence for
    # the source/report it retained. The legacy `recorded` list has no scope.
    lanes = audit_lanes(audit)
    snapshot = audit.get("reference_evidence")
    if not snapshot:
        return unknown_references(lanes, "audit has no retained reference evidence snapshot")
    # A row added after the snapshot still lacks evidence, even when neighboring
    # rows have retained parts. Do not omit it from the summary denominator.
    missing = set(lanes) - snapshot["lanes"].keys()
    if missing:
        return {**snapshot, "lanes": {
            **snapshot["lanes"],
            **unknown_references(missing, "lane absent from retained reference snapshot")["lanes"]}}
    return snapshot


def render_markdown(audit, path):
    nat, py = audit["native"], audit["python"]
    references = audit_references(audit)
    tri = [(r, triage(r), sabotage_of(r)) for r in nat]
    count = defaultdict(int)
    for _, (st, _, _), _ in tri:
        count[st] += 1
    sab = defaultdict(int)
    for _, (st, _, _), (ss, _) in tri:
        if not st.startswith("n/a"):
            sab[ss] += 1
    lines = []
    w = lines.append
    stamp = re.search(r"(\d{4})(\d{2})(\d{2})", os.path.basename(path))
    day = "-".join(stamp.groups()) if stamp else "2026-10-21"
    whole = audit.get("scope") == "all"
    if whole:
        w(f"# Verifier coverage of the whole public surface ({day})")
        w("")
        w(f"Release gate (Andrew, 2026-10-21, widened 2026-10-24 to every exposed API): every native entry "
          f"the package registers and every public operation method is reached by a verifier lane and by a "
          f"fault-injection arm the verifier or a check provably catches. Every row at `{audit['head']}`, "
          f"changed or not (`--all`). Generated by `tools/verify_coverage_audit.py` "
          f"(static: the export walk of `tools/public_runtime_inventory.py`, `tools/lane_select.py`'s Mojo "
          f"definition graph, and a scoped name closure from each lane's own code); the rows marked by hand are "
          f"the `ANNOTATIONS` and `ANNOTATIONS_BY_SRC` tables in that file, each checked in the source.")
    else:
        w(f"# Verifier coverage of the endpoints changed since the last release ({day})")
        w("")
        w(f"Release gate (Andrew, 2026-10-21): every endpoint that is new or changed since the last published "
          f"release is reached by a verifier lane and by a fault-injection arm the verifier or a check provably "
          f"catches. Range `{audit['base']}..{audit['head']}`. Generated by `tools/verify_coverage_audit.py` "
          f"(static: the export walk of `tools/public_runtime_inventory.py`, `tools/lane_select.py`'s Mojo "
          f"definition graph, and a scoped name closure from each lane's own code); the rows marked by hand are "
          f"the `ANNOTATIONS` and `ANNOTATIONS_BY_SRC` tables in that file, each checked in the source.")
    w("")
    w("Regenerate:")
    w("")
    w("```sh")
    w(f"python3 tools/verify_coverage_audit.py {'--all ' if whole else ''}--base {audit['base']} --json "
      f"<audit.json> --markdown docs/plans/{os.path.basename(path)}")
    w("```")
    w("")
    w("## Summary")
    w("")
    w("Native entries (one row per registration; a GPU binding and its host twin are two rows):")
    w("")
    w("| status | rows |")
    w("|---|---|")
    for st in ("covered", "covered (new lane)", "n/a", "owed", "GAP"):
        w(f"| {st} | {count.get(st, 0)} |")
    w(f"| total | {len(nat)} |")
    w("")
    states = defaultdict(int)
    for r in nat:
        states[reach_state(r)] += 1
    w("How each row is reached (precise = a lane provably calls the export; `declared` rows are "
      "`tools/identity_lanes/coverage_declared.tsv` and the hand annotations, each followed in the source; "
      "`wide only` = a lane runs the binding whole and nothing decided the row):")
    w("")
    w("| reach | rows |")
    w("|---|---|")
    for st, label_ in (("static", "precise, static reach"), ("declared", "precise, declared"),
                       ("decided", "decided n/a or dead"), ("wide", "wide only"), ("none", "none")):
        w(f"| {label_} | {states.get(st, 0)} |")
    w("")
    w("Fault-injection arms, over the rows that are not n/a:")
    w("")
    w("| arm | rows |")
    w("|---|---|")
    for ss in ("armed", "n/a", "owed"):
        w(f"| {ss} | {sab.get(ss, 0)} |")
    w("")
    pst = defaultdict(int)
    prow = []
    by_entry = native_status_by_entry(nat)
    for r in py:
        st, lanes, why = py_triage(r, by_entry)
        pst[st] += 1
        prow.append((r, st, lanes, why))
    w(f"Public Python methods{' (every public operation method)' if audit.get('scope') == 'all' else ' whose call path changed'}: {len(py)} "
      + ", ".join(f"{k} {v}" for k, v in sorted(pst.items())) + ".")
    w("")
    w("`covered` describes static reach only. It does not mean that a lane was run, recorded or admitted. "
      "Fault arms marked `armed` likewise describe source reach, not successful fault detection.")
    w("")
    w("## Reach rules (2026-10-24)")
    w("")
    w("- **Native owners.** A migrated public method is one owner call (`xseq_native_call`, "
      "`decomp_native_call`); the owner picks its handler on the operation and family strings and calls "
      "compiled entries by name (`entries[\"croston\"]`, `self.fn(\"x_decomp_dev_gemm\")`, `\"xn_\" + name`). "
      "`OwnerDispatch` reads every `if`/`elif` on a dispatch variable in the Python-facing Mojo, and the "
      "entries its branch and the definitions it calls name (four calls deep, a type's methods followed by "
      "name). A lane reaches those entries when its string constants or class names satisfy every enclosing "
      "condition with a specific value (`croston_fit`, `FastICA`); a branch on generic values only (`fit`, a "
      "package method name, a plain word) is credited only to lanes that reach that owner entry itself, "
      "whose own impl file holds the branch. A branch that names another owner operation relays it.")
    w("- **Bridge connects.** `_IPCA_CALL, _ = _native_ipca_bridge.connect_public(IncrementalPCA)`: reading "
      "the bound name follows the connect and the call's own string arguments (`\"als_native_call\"`).")
    w("- **Python shell.** A public method is API shell (argument checks routing to a native call; the "
      "no-Python-compute fence keeps arithmetic out of it). It is covered when a lane reaches it, or, as "
      "`covered (via native)`, when every native entry its walked call path reaches is covered. A method "
      "with no native work at all (`return self`) is n/a.")
    w("- **Constant probes.** A registration whose whole body returns one compiled constant (a contract "
      "version, vendor or capability flag) reads no data and computes nothing: n/a.")
    w("")
    w("## Retained reference evidence")
    w("")
    w("RETAINED means that numeric reference parts have linked record metadata in the captured reference "
      "table. It does not establish complete fixtures, all vendor columns, installed-package execution, "
      "source applicability or admission of this candidate. The JSON preserves the actual parts, columns "
      "and source records, including observed values that differ from the saved reference. "
      "RETAINED does not mean those columns agree. Intentionally partial records remain partial. OWED means no numeric reference "
      "is retained; UNKNOWN means the evidence is unavailable or inconsistent. Candidate admission: "
      "**not assessed**.")
    w("")
    if references.get("table_sha256"):
        w(f"Reference table SHA-256: `{references['table_sha256']}`.")
        w("")
    w("| reference evidence | lanes |")
    w("|---|---|")
    for status in ("RETAINED", "OWED", "UNKNOWN"):
        selected = sorted(lane for lane, item in references["lanes"].items()
                          if item["status"] == status)
        w(f"| {status} | {len(selected)} |")
    w("")
    comparisons = references.get("column_comparisons")
    if comparisons is not None:
        w(f"Linked recorded column values: {comparisons['equal']} equal to their saved reference, "
          f"{comparisons['different']} different, and {comparisons['n/a']} explicitly N/A. "
          "These are historical observations, not current-candidate identity or admission results.")
        w("")
        different = sorted((lane, item["differing_columns"]) for lane, item in references["lanes"].items()
                           if item.get("differing_columns"))
        if different:
            w("| lane with differing recorded values | column counts |")
            w("|---|---|")
            for lane, columns in different:
                w(f"| `{lane}` | " + ", ".join(f"{column} {count}" for column, count in sorted(columns.items())) + " |")
            w("")
    for status in ("OWED", "UNKNOWN"):
        selected = sorted(lane for lane, item in references["lanes"].items()
                          if item["status"] == status)
        if selected:
            w(f"{status} lanes:")
            w("")
            for lane in selected:
                w(f"- `{lane}`: {references['lanes'][lane]['reason']}")
            w("")
    w("")
    proven = _proven_arms()
    w("## Fault-injection arms added (tools/identity_lanes/coverage_sabotage.tsv)")
    w("")
    w("One source patch per changed kernel file that no arm reached, each a value fault in the changed "
      "definitions (a float one ulp off, an index permuted, a byte flipped), applied only to a scratch copy. "
      "`python3 tools/coverage_sabotage.py plan --arm <arm>` describes clean, sabotaged and restored runs. "
      "`check` requires matching source/device/input metadata, complete fixtures, valid clean results, "
      "fault detection in every named lane, and restoration to clean output. Refusals require an exact "
      "fault-specific review; missing bindings and other infrastructure failures do not count. "
      + ("Retained execution scope is listed under Arms proven below; a declared arm is not proof it ran." if proven else
         "RUN OWED: no execution evidence is available in the retained arms table. This static audit does not "
         "run apply-check or fault qualification."))
    w("")
    try:
        import coverage_sabotage
        table = coverage_sabotage.arms()
    except (ImportError, OSError, SystemExit):
        table = {}
    if table:
        w("| arm | patch | bindings | lanes |")
        w("|---|---|---|---|")
        for name, arm in table.items():
            w(f"| `{name}` | `{arm['patch']}` | {', '.join(arm['bindings'])} | {_short(arm['lanes'], 4)} |")
    else:
        w("(none yet)")
    w("")
    if proven:
        w("## Arms proven (tools/identity_lanes/coverage_sabotage_proven.tsv)")
        w("")
        w("The following is retained historical evidence. This static report does not establish its "
          "applicability to the candidate's source, binding ownership or physical library layout.")
        w("")
        w("Run one at a time on the Mac (lane run-sabotage-arms, 2026-10-21) over the merged Metal (identical) "
          "and host builds of 51bf10f4f, whose kernel sources equal 79982adaa's. Per arm: the patch applied, the "
          "named per-binding builds rebuilt (a binding's own file wins over the merged one), the arm's lanes run "
          "(`identity_break.py --fixtures base --repeats 1`) on Metal and on the host column "
          "(`MOJOLEARN_VENDOR=cpu`) for the columns whose bindings were rebuilt, then the patch reverted and its "
          "builds removed and the lanes run again. A lane is caught when a column it runs on MOVED or REFUSED "
          "against the clean run of the same lanes; the sabotaged cells are also judged against "
          "`python/mojolearn/verify_reference/table.json` (OWED: a new lane with no reference yet, judged "
          "against the clean run only; IDENTICAL: a column the fault does not reach while the other column "
          "moved). Every REFUSED cell is the fault tripping one of the library's own checks.")
        w("")
        w("| arm | builds (rc, seconds) | verdict | lanes (column: result) | sabotaged cells against table.json "
          "| patch reverted | seconds |")
        w("|---|---|---|---|---|---|---|")
        for row in proven:
            w("| `" + row[0] + "` | " + " | ".join(row[1:]) + " |")
        w("")
        caught = sum(1 for row in proven if row[2] == "CAUGHT")
        w(f"{caught} of {len(proven)} arms caught; {sum(int(row[6]) for row in proven)} arm seconds in all.")
        w("")
        for line in _PROVEN_FIXES:
            w(line)
        w("")
        w("### Arm decisions after the 2026-10-24 run (9b04b92c5: 12 CAUGHT, 9 MISSED)")
        w("")
        for line in _ARM_DECISIONS_20261024:
            w(line)
        w("")
    w("## Owed")
    w("")
    owed = [(r, t) for r, t, _ in tri if t[0] == "owed"]
    for why in sorted({t[2] for _, t in owed}):
        names = sorted({r["entry"] for r, t in owed if t[2] == why})
        w(f"- {', '.join(f'`{n}`' for n in names)}: {why}")
    gaps = [r for r, t, _ in tri if t[0] == "GAP"]
    if not owed and not gaps:
        w("Native entries: none.")
    if gaps:
        w(f"- GAP (no lane, no reason): {', '.join(sorted({r['entry'] for r in gaps}))}")
    sab_owed = defaultdict(set)
    for r, t, (ss, txt) in tri:
        if ss == "owed" and not t[0].startswith("n/a"):
            sab_owed[txt].add(r["entry"])
    if not sab_owed:
        w("")
        w("Sabotage arms: none.")
    if sab_owed:
        w("")
        w("Sabotage arms still owed:")
        w("")
        for why in sorted(sab_owed):
            names = sorted(sab_owed[why])
            w(f"- " + ", ".join(f"`{n}`" for n in names[:10]) + (f" +{len(names) - 10}" if len(names) > 10 else "")
              + f": {why}")
    dead_rows = sorted((r["entry"], r["src"], t[2][len(DEAD_PREFIX):]) for r, t, _ in tri
                       if t[0] == "n/a" and t[2].startswith(DEAD_PREFIX))
    w("")
    w("## Dead entries (listed for deletion)")
    w("")
    if not dead_rows:
        w("None.")
    else:
        w(f"{len(dead_rows)} registrations with no public path (no package code, native owner or lane calls "
          f"them; `dead` in `tools/identity_lanes/coverage_declared.tsv`). Each is a deletion candidate: the "
          f"export, and its definitions when nothing else reaches them.")
        w("")
        w("| entry | binding | who names it |")
        w("|---|---|---|")
        for entry, src, why in dead_rows:
            w(f"| `{entry}` | `{os.path.basename(src)}` | {why} |")
    w("")
    w("## Method and limits")
    w("")
    w("- A registration is CHANGED when a Mojo definition its export reaches (lane_select's definition graph, "
      "by name through imports) differs from the base, moves excluded (the same text at the base under the same "
      "name in any file). A shared helper's change therefore marks every export that reaches it.")
    w("- A lane REACHES an entry when the entry's name is in the lane's scoped closure: the lane function, its "
      "helpers and property probes (the harness's shared helpers excluded), the package definitions those name "
      "(resolved through imports, methods of reached classes by attribute name, import-time calls, module tables, "
      "`name + \"_suffix\"` composition), or the public method walk of tools/public_runtime_inventory.py. It is "
      "static: a reach is a call path, not a run. New and existing lanes require separately retained execution, "
      "reference and installed-package records for each claimed target and option; this table does not admit them.")
    w("- A row is ARMED when a definition in its export's closure reads a `MOJOLEARN_*SABOTAGE*` define or is "
      "edited by a `*/sabotage/*.patch`. `CPU gate HOST_SABOTAGE` is the arm the CPU identity gate builds and "
      "requires to move every covered part; family defines and seam patches are their checks' arms; coverage arms "
      "are this audit's (" + ("run: Arms proven" if proven else "RUN OWED") + "). Definitions with no value to fault are listed with the reason in "
      "tools/identity_lanes/coverage_sabotage_na.tsv.")
    w("- `n/a` lanes: no Python caller in the package, the fast tier (the verifier refuses fast), a verifier or "
      "build probe, or a dead helper; each with its reason in the row.")
    w("")
    w("## Public Python methods")
    w("")
    w("| method | where | lanes | static reach | reference evidence |")
    w("|---|---|---|---|---|")
    for r, st, lanes, why in prow:
        w(f"| `{r['public']}.{r['op']}` | `{r['where']}` | {_short(lanes)} | {st}{': ' + why if why else ''} "
          f"| {reference_label(lanes, references)} |")
    w("")
    w("## Native entries")
    w("")
    w("Change: `new` (registered nowhere at the base), `door`, `impl` or `kernel` (the kinds of the changed "
      "Mojo definitions the export reaches). Sabotage: `CPU gate HOST_SABOTAGE` is the verifier's own "
      "negative control (every host binding rebuilt with it, every covered lane's parts must move); other "
      "defines and seam patches are the family check arms.")
    by_src = defaultdict(list)
    for r, t, sb in tri:
        by_src[r["src"]].append((r, t, sb))
    for src in sorted(by_src):
        w("")
        w(f"### `{src}`")
        w("")
        w("| entry | change | lanes | sabotage | static reach | reference evidence |")
        w("|---|---|---|---|---|---|")
        for r, (st, lanes, why), (ss, txt) in sorted(by_src[src], key=lambda x: x[0]["entry"]):
            if st.startswith("n/a") and ss != "armed":
                ss, txt = "n/a", "no lane can run it (see status)"
            change = "new" if r["new_anywhere"] else ("moved here" if r["new"] else "/".join(r["kinds"]))
            w(f"| `{r['entry']}` | {change} | {_short(lanes)} | {ss}: {txt} | {st}{': ' + why if why else ''} "
              f"| {reference_label(lanes, references)} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return count, sab, pst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="fc8d425fb")
    ap.add_argument("--json", help="write the audit here")
    ap.add_argument("--from-json", help="read a previous audit instead of deriving one")
    ap.add_argument("--markdown", help="write the coverage table here")
    ap.add_argument("--all", dest="whole_surface", action="store_true",
                    help="audit the WHOLE public surface: every registration and every public "
                         "operation method, not only those new or changed since --base")
    args = ap.parse_args(argv)
    if args.from_json:
        audit = json.load(open(args.from_json))
    else:
        import lane_select as ls
        walker = Walker()
        reach = lane_reach(ls, walker, NameClosure(walker))
        reach, _ = owner_reach(ls, reach)
        native = native_rows(ls, args.base, reach, args.whole_surface)
        py = python_rows(ls, walker, reach, args.base, args.whole_surface)
        audit = dict(base=args.base, head=git("rev-parse", "HEAD").strip(),
                     scope="all" if args.whole_surface else "changed",
                     lane_inventory=sorted(reach), native=native, python=py)
        audit['endpoints'] = endpoint_rows(walker, reach)
        audit["reference_evidence"] = reference_snapshot(
            os.path.join(ROOT, "python/mojolearn/verify_reference/table.json"), audit_lanes(audit))
        # Compatibility field: scoped retained metadata, never full admission.
        audit["recorded"] = sorted(lane for lane, item in audit["reference_evidence"]["lanes"].items()
                                   if item["status"] == "RETAINED")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(audit, fh, indent=1, sort_keys=True)
    if args.markdown:
        count, sab, pst = render_markdown(audit, args.markdown)
        print("native status: " + ", ".join(f"{k} {v}" for k, v in sorted(count.items())))
        print("sabotage: " + ", ".join(f"{k} {v}" for k, v in sorted(sab.items())))
        print("python: " + ", ".join(f"{k} {v}" for k, v in sorted(pst.items())))
    references = audit_references(audit)
    nat = audit["native"]
    states = defaultdict(int)
    dead = 0
    for r in nat:
        states[reach_state(r)] += 1
        dead += triage(r)[2].startswith(DEAD_PREFIX)
    unknown = sorted({lane for d in declared().values() for lane in d[1]} - set(audit.get("lane_inventory", ())))
    if unknown and audit.get("lane_inventory"):
        print(f"DECLARED LANES NOT IN THE INVENTORY ({DECLARED_TSV}): " + ", ".join(unknown[:20]))
    print(f"native: {len(nat)} new/changed registrations; "
          f"{states['static'] + states['declared']} precise lane(s) ({states['static']} static, "
          f"{states['declared']} declared), "
          f"{states['wide']} wide only, "
          f"{states['decided']} decided n/a ({dead} dead, listed for deletion), "
          f"{states['none']} none; "
          f"{sum(1 for r in nat if r['sabotage_defines'] or r['sabotage_patches'])} with a sabotage arm")
    print(f"python: {len(audit['python'])} changed public methods; "
          f"{sum(1 for r in audit['python'] if r['lanes'])} with a lane")
    endpoint_counts = defaultdict(int)
    for row in audit.get('endpoints', ()):
        endpoint_counts[row['status']] += 1
    print('complete public endpoint census (source mapping only): ' +
          ', '.join(f'{key} {value}' for key, value in sorted(endpoint_counts.items())))
    print("reference evidence: " + reference_label(audit_lanes(audit), references))
    print("Static audit completed; execution and candidate admission not assessed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
