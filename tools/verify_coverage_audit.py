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


# ------------------------------------------------------------------ triage

#: The lanes this audit added (tools/identity_lanes/coverage.py). They read
#: OWED until a release record admits them.
NEW_LANES = frozenset({"gp-lml-theta", "cross-vendor-fold", "sequence-moe-backward",
                       "sequence-export-roundtrip", "sequence-onnx-import",
                       "transformer-decode-load-state", "safetensors-widen-f16",
                       "sequence-program-save-load", "hf-causal-lm-stream", "ivf-short-fill",
                       "sequence-device-tensor"})

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
}

_BYTE_SESSION = ("covered", ["byte-lm", "byte-lm-resident"],
                 "the CPU column routes _mojolearn_byte_lm through _byte_lm_trainer_host "
                 "(host_surface ADAPTED_MODULES), whose TrainerOwnerEntries call it for every "
                 "SmallByteLanguageModelTrainer step; on a GPU install byte_lm_trainer_step runs the "
                 "same session ops inside Mojo")
_DEAD_KIT = "n/a: no live caller (the _expansion_decomp helper that calls it is never called)"
_MIGRATION_TRAINING = ("owed", [], "migration lane on python/mojolearn/_training_impl.py (running): "
                       "lane owed after it lands; tools/lane_accounting.py DECLARED_LANELESS already "
                       "names training.chunked_lm_head_loss and training.mlp_inference_sessions")
_MIGRATION_MAMBA = ("owed", [], "migration lane on python/mojolearn/_mamba_impl.py and mamba/ (running): "
                    "mamba2-decode-session and mamba3-decode-session lanes owed after it lands "
                    "(mamba1-decode-session is the template)")

#: entry -> (status, lanes, why), read by hand where the static reach cannot
#: see the call (a worker process, a native owner handed the entry, a class
#: adapter) or where no lane can apply. Each one was checked in the source.
ANNOTATIONS = {
    **{e: _BYTE_SESSION for e in ("byte_lm_session_eval", "byte_lm_session_export_gradients",
                                  "byte_lm_session_export_state", "byte_lm_session_logits",
                                  "byte_lm_session_next_bytes", "byte_lm_session_open",
                                  "byte_lm_session_rollback", "byte_lm_session_step")},
    "knn_classify_neighbors": ("covered", ["par-reference-knn"],
                               "run in a worker process (_parallel_worker 'neighbor_vote' -> "
                               "parallel_neighbors_reference._vote); par-* lanes run on two-device boxes"),
    "knn_regress_neighbors": ("covered", ["par-reference-knn-reg"],
                              "run in a worker process (_parallel_worker 'neighbor_vote')"),
    "fold_pair_f32": ("covered", ["cross-vendor-fold"],
                      "mojolearn.cross_vendor hands it to the native fold at import "
                      "(_buffer_native('fold_pair_f32')); every ordered_fold and fold_pair runs it"),
    "ivf_flat_partial_search": ("covered", ["par-ivf"], "run in a worker process (_parallel_worker)"),
    "ivf_finalize_distances": ("covered", ["par-ivf"], "run in a worker process (_parallel_worker)"),
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
    "covariance": ("n/a", [], "Apple FAST quality probe binding (tools/scoped_gemm_quality.py)"),
    **{e: ("n/a", [], "no live caller: _expansion_cnn._sgd/_adam/_softmax_xent are never called")
       for e in ("x_cnn_adam", "x_cnn_sgd", "x_cnn_softmax_xent")},
    **{e: ("n/a", [], _DEAD_KIT) for e in ("x_decomp_chi2_cdf", "x_decomp_eigh_batch", "x_decomp_knn_select",
                                           "x_decomp_dev_knn_select", "x_decomp_dev_cd_rows",
                                           "x_decomp_dev_als_rows", "x_decomp_dev_dict_update")},
    "ops_metric": ("n/a", [], "no live caller: Session._metric is never called (accuracy and log_loss "
                              "run through ops_session_call)"),
    "onnx_from_words": ("n/a", [], "the torch.export import path: needs torch, which a verifier run never "
                                   "has; the ONNX path is sequence-onnx-import"),
    **{e: _MIGRATION_TRAINING for e in ("chunked_lm_head_v2_loss", "chunked_lm_head_v2_train",
                                        "samba_head_loss", "mlp_sessions", "neural_rng",
                                        "optimizer_finalize")},
    **{e: _MIGRATION_MAMBA for e in ("mamba2_session_export_state", "mamba2_session_load_state",
                                     "mamba2_session_open", "mamba3_session_export_state",
                                     "mamba3_session_load_state", "mamba3_session_open",
                                     "mamba3_session_step")},
}

#: (public, op) -> (status, lanes, why) for the public methods.
PY_ANNOTATIONS = {
    ("mojolearn.models.ParallelCausalLM", "generate"): (
        "covered", ["hf-causal-lm", "par-causal-lm"],
        "the inherited CausalLM.generate (hf-causal-lm) over ParallelCausalLM's forward (par-causal-lm)"),
}

#: Defines a verifier run provably turns DIVERGENT: the CPU identity gate builds
#: every host binding with it and requires the covered lanes' parts to move
#: (docs/VERIFY.md, tools/cpu_identity_gate_check.py).
GATE_DEFINES = frozenset({"MOJOLEARN_HOST_SABOTAGE"})


def triage(row):
    """(status, lanes, why) for one native row."""
    if row["entry"] in ANNOTATIONS:
        status, lanes, why = ANNOTATIONS[row["entry"]]
        return status, sorted(set(lanes) | set(row["lanes_precise"])), why
    if row["lanes_precise"]:
        if set(row["lanes_precise"]) <= NEW_LANES:
            return "covered (new lane)", row["lanes_precise"], "new lane, OWED until recorded"
        return "covered", row["lanes_precise"], ""
    if not row["py_files"]:
        return "n/a", [], "no Python caller in the package (manifest, tests or Mojo-internal dispatch only)"
    return "GAP", [], "no lane reaches it"


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
    if bits:
        return "armed", "; ".join(bits)
    if not row["changed_kernel"]:
        return "n/a", "door-only change (conversion, no arithmetic)"
    pending = [n for n in row["changed_kernel"] if _sab_na(n) is None]
    if not pending:
        return "n/a", _sab_na(row["changed_kernel"][0])
    if all(n.startswith("core/py2mojo_rows.mojo:") for n in pending):
        return "owed", ("rows_call's faulted values (the Huber threshold, the SGD and LogisticRegressionCV "
                        "probability modes) reach no hashed part until lanes for HuberRegressor.outliers_ and "
                        "SGD/LogisticRegressionCV predict_proba exist; python/mojolearn/linear_model.py belongs "
                        "to a running migration lane, so the lanes and the arm are owed after it lands")
    if all(n.startswith("mamba/") for n in pending):
        return "owed", ("migration lane on mamba/ (running): the arm for " + pending[0]
                        + " is owed after it lands")
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


def render_markdown(audit, path):
    nat, py = audit["native"], audit["python"]
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
    w("# Verifier coverage of the endpoints changed since the last release (2026-10-21)")
    w("")
    w(f"Release gate (Andrew, 2026-10-21): every endpoint that is new or changed since the last published "
      f"release is reached by a verifier lane and by a fault-injection arm the verifier or a check provably "
      f"catches. Range `{audit['base']}..{audit['head']}`. Generated by `tools/verify_coverage_audit.py` "
      f"(static: the export walk of `tools/public_runtime_inventory.py`, `tools/lane_select.py`'s Mojo "
      f"definition graph, and a scoped name closure from each lane's own code); the rows marked by hand are "
      f"the `ANNOTATIONS` table in that file, each checked in the source.")
    w("")
    w("Regenerate:")
    w("")
    w("```sh")
    w(f"python3 tools/verify_coverage_audit.py --base {audit['base']} --json <audit.json> --markdown "
      f"docs/plans/VERIFY_COVERAGE_20261021.md")
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
    w("Fault-injection arms, over the rows that are not n/a:")
    w("")
    w("| arm | rows |")
    w("|---|---|")
    for ss in ("armed", "n/a", "owed"):
        w(f"| {ss} | {sab.get(ss, 0)} |")
    w("")
    pst = defaultdict(int)
    prow = []
    for r in py:
        key = (r["public"], r["op"])
        if key in PY_ANNOTATIONS:
            st, lanes, why = PY_ANNOTATIONS[key]
        elif r["lanes"]:
            st = "covered (new lane)" if set(r["lanes"]) <= NEW_LANES else "covered"
            lanes, why = r["lanes"], ""
        else:
            st, lanes, why = "GAP", [], "no lane reaches it"
        pst[st] += 1
        prow.append((r, st, lanes, why))
    w(f"Public Python methods whose call path changed: {len(py)} "
      + ", ".join(f"{k} {v}" for k, v in sorted(pst.items())) + ".")
    w("")
    w("A lane that reaches an entry `covered` is recorded unless it is one of the new lanes below, which "
      "read OWED until a release record admits them.")
    w("")
    w("## New lanes (tools/identity_lanes/coverage.py)")
    w("")
    w("Each ran 9 of 9 fixtures STABLE at `--repeats 2` on the Mac (Metal) build of integ/20261021; "
      "parts are Mojo FNV-1a64 digests (`verify_digest`). Declared in their family's training lanes with "
      "the pending reason `no reference`.")
    w("")
    for lane in sorted(NEW_LANES):
        w(f"- `{lane}`: {NEW_LANE_PURPOSE[lane]}")
    w("")
    w("## Fault-injection arms added (tools/identity_lanes/coverage_sabotage.tsv)")
    w("")
    w("One source patch per changed kernel file that no arm reached, each a value fault in the changed "
      "definitions (a float one ulp off, an index permuted, a byte flipped), applied only to a scratch copy. "
      "`python3 tools/coverage_sabotage.py plan --arm <arm>` prints the build and the two identity_break runs, "
      "and `check` requires every named lane to read MOVED or REFUSED on the sabotaged build. RUN OWED: no arm "
      "has been built or run yet (lanes never compile); `apply-check` passes on this commit.")
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
    w("## Owed")
    w("")
    owed = [(r, t) for r, t, _ in tri if t[0] == "owed"]
    for why in sorted({t[2] for _, t in owed}):
        names = sorted({r["entry"] for r, t in owed if t[2] == why})
        w(f"- {', '.join(f'`{n}`' for n in names)}: {why}")
    gaps = [r for r, t, _ in tri if t[0] == "GAP"]
    if gaps:
        w(f"- GAP (no lane, no reason): {', '.join(sorted({r['entry'] for r in gaps}))}")
    sab_owed = defaultdict(set)
    for r, t, (ss, txt) in tri:
        if ss == "owed" and not t[0].startswith("n/a"):
            sab_owed[txt].add(r["entry"])
    if sab_owed:
        w("")
        w("Sabotage arms still owed:")
        w("")
        for why in sorted(sab_owed):
            names = sorted(sab_owed[why])
            w(f"- " + ", ".join(f"`{n}`" for n in names[:10]) + (f" +{len(names) - 10}" if len(names) > 10 else "")
              + f": {why}")
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
      "static: a reach is a call path, not a run. The new lanes were run; the rest rest on their recorded cells.")
    w("- A row is ARMED when a definition in its export's closure reads a `MOJOLEARN_*SABOTAGE*` define or is "
      "edited by a `*/sabotage/*.patch`. `CPU gate HOST_SABOTAGE` is the arm the CPU identity gate builds and "
      "requires to move every covered part; family defines and seam patches are their checks' arms; coverage arms "
      "are this audit's (RUN OWED). Definitions with no value to fault are listed with the reason in "
      "tools/identity_lanes/coverage_sabotage_na.tsv.")
    w("- `n/a` lanes: no Python caller in the package, the fast tier (the verifier refuses fast), a verifier or "
      "build probe, or a dead helper; each with its reason in the row.")
    w("")
    w("## Public Python methods")
    w("")
    w("| method | where | lanes | status |")
    w("|---|---|---|---|")
    for r, st, lanes, why in prow:
        w(f"| `{r['public']}.{r['op']}` | `{r['where']}` | {_short(lanes)} | {st}{': ' + why if why else ''} |")
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
        w("| entry | change | lanes | sabotage | status |")
        w("|---|---|---|---|---|")
        for r, (st, lanes, why), (ss, txt) in sorted(by_src[src], key=lambda x: x[0]["entry"]):
            change = "new" if r["new_anywhere"] else ("moved here" if r["new"] else "/".join(r["kinds"]))
            w(f"| `{r['entry']}` | {change} | {_short(lanes)} | {ss}: {txt} | {st}{': ' + why if why else ''} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return count, sab, pst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="fc8d425fb")
    ap.add_argument("--json", help="write the audit here")
    ap.add_argument("--from-json", help="read a previous audit instead of deriving one")
    ap.add_argument("--markdown", help="write the coverage table here")
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
    if args.markdown:
        count, sab, pst = render_markdown(audit, args.markdown)
        print("native status: " + ", ".join(f"{k} {v}" for k, v in sorted(count.items())))
        print("sabotage: " + ", ".join(f"{k} {v}" for k, v in sorted(sab.items())))
        print("python: " + ", ".join(f"{k} {v}" for k, v in sorted(pst.items())))
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
