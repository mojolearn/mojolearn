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
    files = [p for p in git("diff", "--name-only", f"{base}..HEAD", "--", "*.mojo").split() if p]
    status = {}
    for line in git("diff", "--name-status", f"{base}..HEAD", "--", "*.mojo").splitlines():
        parts = line.split("\t")
        status[parts[-1]] = parts[0]
    old_texts = defaultdict(set)          # unit key without file -> texts at BASE
    for path in files:
        old = show(base, path)
        if old is None:
            continue
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
            precise = {lane for lane, (_, ents) in reach.items() if export in ents}
            rows.append(dict(entry=export, src=src, impl=impl, new=new, unresolved=not roots,
                             kinds=kinds, changed=[label(n) for n in hits[:4]], n_changed=len(hits),
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


def lane_reach(ls, walker):
    """lane -> (set of (public key, op) roots, set of entry names reached)."""
    ib = ls.identity_break()
    _, why_map = ls.lane_sources()
    out = {}
    for lane, fn in ib.LANES.items():
        try:
            names = ls._seed_names(fn, fn.__globals__)
        except Exception:
            names = set()
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
        out[lane] = (roots, entries)
    return out


def python_rows(ls, walker, reach, base):
    pri = walker.pri
    index = walker.index
    changed = changed_python_lines(base)
    root_lanes = defaultdict(set)
    for lane, (roots, _) in reach.items():
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
        reach = lane_reach(ls, walker)
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
