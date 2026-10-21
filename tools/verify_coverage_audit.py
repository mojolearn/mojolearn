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

REG_RE = re.compile(
    r"def_(?:py_c_|py_)?function\[\s*([A-Za-z0-9_]+)\s*(?:\[([^\[\]]*)\])?\s*\]\s*\(\s*\"([A-Za-z0-9_]+)\"")
SAB_RE = re.compile(r"MOJOLEARN_[A-Z0-9_]*SABOTAGE[A-Z0-9_]*")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=False).stdout


def show(rev, path):
    out = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{path}"], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def registrations(text):
    """name -> (impl, [param identifiers]) for one binding source."""
    out = {}
    for m in REG_RE.finditer(text or ""):
        out[m.group(3)] = (m.group(1), IDENT_RE.findall(m.group(2) or ""))
    return out


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
        target, removed = None, defaultdict(list)
        for line in text.splitlines():
            if line.startswith("+++ "):
                target = line[4:].strip()
                target = target[2:] if target.startswith("b/") else target
            elif line.startswith("-") and not line.startswith("---") and target:
                body = line[1:].strip()
                if len(body) > 6:
                    removed[target].append(body)
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


def native_rows(ls, base, reach):
    changed, whole, status = changed_mojo_nodes(ls, base)
    pnodes = patch_nodes(ls)
    graph = ls._MojoDefGraph()
    srcs = sorted(p for p in git("ls-files", "bindings/*.mojo", "bindings/**/*.mojo").split())
    py_words = defaultdict(int)
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
            for w in words:
                py_words[w] += 1
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
            if not new and not hits:
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
            rows.append(dict(entry=export, src=src, impl=impl, new=new, new_anywhere=export not in anywhere, py_files=py_words.get(export, 0), unresolved=not roots,
                             kinds=kinds, changed=[label(n) for n in hits[:4]], n_changed=len(hits),
                             changed_kernel=[label(n) for n in hits if kind_of(n[1]) != "door"],
                             lanes_precise=sorted(precise), lanes_wide=sorted(set(users) - precise),
                             lanes_closure=len(users),
                             sabotage_defines=sorted(sab_defs), sabotage_sites=sorted(sab_sites)[:4],
                             sabotage_patches=sorted(patches)))
    return rows


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
        for op in set(self.pri.OPS) | {"__init__"}:
            if op not in names and op not in HARNESS_OPS and op != "__init__":
                continue
            found = self.index.lookup(cls, op)
            if found and found[0] == "func":
                out.append((op, found[1]))
        return out


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
        self._facts = {}

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
        defs, names = [], set()
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
        reached = set(seeds) | HARNESS_OPS
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
        reached = closer.closure(symbols) if closer is not None else set()
        entries |= reached
        out[lane] = (roots, entries, reached)
    return out


def python_rows(ls, walker, reach, base):
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
            if not path_changed:
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
                             helpers_changed=[f"{os.path.relpath(str(f.mod.path), ROOT)}:{f.qual}"
                                              for f in path_changed[:3]],
                             n_helpers_changed=len(path_changed), entries=sorted(entries),
                             lanes=sorted(lanes)))
    rows.sort(key=lambda r: (r["public"], r["op"]))
    return rows


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="fc8d425fb")
    ap.add_argument("--json", help="write the audit here")
    ap.add_argument("--from-json", help="read a previous audit instead of deriving one")
    args = ap.parse_args(argv)
    if args.from_json:
        audit = json.load(open(args.from_json))
    else:
        import lane_select as ls
        walker = Walker()
        reach = lane_reach(ls, walker, NameClosure(walker))
        native = native_rows(ls, args.base, reach)
        py = python_rows(ls, walker, reach, args.base)
        audit = dict(base=args.base, head=git("rev-parse", "--short", "HEAD").strip(),
                     recorded=sorted(recorded_lanes()), native=native, python=py)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(audit, fh, indent=1, sort_keys=True)
    rec = set(audit["recorded"])
    nat = audit["native"]
    print(f"native: {len(nat)} new/changed registrations; "
          f"{sum(1 for r in nat if r['lanes_precise'])} precise lane(s), "
          f"{sum(1 for r in nat if not r['lanes_precise'] and r['lanes_wide'])} wide only, "
          f"{sum(1 for r in nat if not r['lanes_precise'] and not r['lanes_wide'])} none; "
          f"{sum(1 for r in nat if r['sabotage_defines'] or r['sabotage_patches'])} with a sabotage arm")
    print(f"python: {len(audit['python'])} changed public methods; "
          f"{sum(1 for r in audit['python'] if r['lanes'])} with a lane; recorded lanes {len(rec)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
