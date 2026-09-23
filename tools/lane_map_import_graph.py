#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LANE MAP FROM THE IMPORT GRAPH ALONE (handoff 2026-09-22, item 3).

`tools/lane_select.py` derives lane -> source files and the light release
default runs only the lanes whose files a change reaches. Its Mojo side
narrows each binding to the exports a lane's door calls and, where nothing
resolves, falls back to what `host_surface.py` DECLARES for the family
(`host_modules`, `training_lanes`, `inference_lanes`, `loaded_by`). A
declaration is a hand-kept map, and a hand-kept map rots silently: the day a
kernel moves to a file the manifest does not list, the short selection reads
"nothing affected" and a full sweep is the only thing that would notice.

This file computes the map that needs no declaration: for every lane, the
files its Python entry points and its bindings TRANSITIVELY IMPORT, read from
the `from ... import` / `import` statements of the .py and .mojo sources.

  Python   the lane body's names -> the package files defining them, closed
           over the package's own imports (`lane_select._python_closure`,
           subpackages resolved, registries included but not followed, the
           subclass edge followed backwards).
  binding  every `_mojolearn_*` name a file of that closure resolves by
           syntax, plus the host binding `_backend` routes it to on a CPU-only
           install (`routed_modules`, `inference_routes`, `ADAPTED_MODULES`:
           runtime tables, read as code).
  Mojo     the binding's build script, read the way the build reads it
           (`bincache.script_plan`: the root .mojo and the `-I` include roots
           on the `mojo build` line), then `bincache.parse_imports` and
           `bincache.resolve` transitively. This is the same walk that keys
           the binding cache and that `binding_stamps.py` digests after every
           build, so a stamp's recorded closure and this map agree by
           construction (`--check-stamps` holds that against the stamps on
           disk).

It knows nothing of exports and takes no family's word: a binding
contributes its WHOLE compiled closure to every lane that loads it, because
that is what the compiler linked into the .so the lane runs. So the map is
wider than the selector's, and that is its use: it is the outer bound the
narrow map must stay inside.

    python3 tools/lane_map_import_graph.py --compare      disagreements against lane_select
    python3 tools/lane_map_import_graph.py --lane kmeans  one lane's derived files
    python3 tools/lane_map_import_graph.py --file cluster/host/kmeans_oracle.mojo
    python3 tools/lane_map_import_graph.py --json out.json
    python3 tools/lane_map_import_graph.py --check-stamps
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bincache      # noqa: E402
import lane_select   # noqa: E402

BINDINGS = "bindings"


def build_scripts():
    """binding name -> the build script that compiles it, from the `mojo
    build` lines under bindings/ (`bincache.script_plan`). `build_host_family.sh`
    is the builder every host shim execs and compiles nothing by itself."""
    out = {}
    for script in sorted(Path(ROOT, BINDINGS).glob("*.sh")):
        if script.name == "build_host_family.sh":
            continue
        plan = bincache.script_plan(Path(ROOT), script, [])
        if plan is None:
            continue
        for root in plan[0]:
            name = root.name[:-5]
            out.setdefault(name, os.path.relpath(script, ROOT))
    return out


def mojo_closure(script):
    """(files, scope) for one build script: the root's transitive import
    closure the way the build resolves it, plus the shell files the script
    execs. `scope` is "closure", or "tree" when an import names this tree and
    nothing matches, which is the build's own widening (the whole tree is
    then a source of the binary, and the map says so)."""
    plan = bincache.script_plan(Path(ROOT), Path(ROOT, script), [])
    files, scope = set(), "closure"
    if plan is None:
        scope = "tree"
        shells = [Path(ROOT, script), Path(ROOT, BINDINGS, "build_host_family.sh")]
    else:
        roots, incs, shells = plan
        todo = list(roots)
        while todo:
            f = todo.pop()
            if f in files:
                continue
            files.add(f)
            for module, names in bincache.parse_imports(f.read_text(errors="replace")):
                got = bincache.resolve(module, names, f, incs)
                if got is None:
                    scope = "tree"
                    todo = []
                    break
                todo.extend(g.resolve() for g in got if g.resolve() not in files)
    if scope == "tree":
        files = set(p.resolve() for p in bincache.tree_sources(ROOT))
    rels = {os.path.relpath(f, ROOT) for f in files}
    rels |= {os.path.relpath(s, ROOT) for s in shells if Path(s).is_file()}
    return rels, scope


_CLOSURES = {}


def binding_closure(name):
    """The files compiled into binding `name`, memoized."""
    if name not in _CLOSURES:
        script = build_scripts().get(name)
        if script is None:
            _CLOSURES[name] = (None, "no build script")
        else:
            _CLOSURES[name] = mojo_closure(script)
    return _CLOSURES[name]


def lane_bindings(doors, hs):
    """The bindings a lane's Python doors load (`lane_select.resolved_bindings`:
    by syntax, plus the `host_model` dispatch for the classes they define),
    and the host bindings a CPU-only install routes each of them to
    (`lane_select.host_routes`, `ADAPTED_MODULES`)."""
    named = lane_select.resolved_bindings(doors)
    routed = lane_select.host_routes(hs)
    adapted = {k: v["family"] for k, v in hs.ADAPTED_MODULES.items()}
    out = set()
    for b in named:
        out.add(b)
        out |= routed.get(b, set())
        if b in adapted:
            out.add(f"_mojolearn_{adapted[b]}_host")
    return {b for b in out if os.path.exists(os.path.join(ROOT, BINDINGS, b + ".mojo"))}


_DERIVED = None


def derive():
    """lane -> files, from the import graph alone, with per-lane evidence."""
    global _DERIVED
    if _DERIVED is not None:
        return _DERIVED
    ib = lane_select.identity_break()
    hs = lane_select.host_surface()
    symbols = lane_select._python_symbols(lane_select._python_files())
    sinks = lane_select.enumerator_files() | lane_select._reexport_registries()
    sources, why = {}, {}
    for lane, fn in ib.LANES.items():
        names = lane_select._seed_names(fn, vars(ib))
        seeds = set()
        for name in names:
            seeds |= set(symbols.get(name, ()))
        py = lane_select._python_closure(seeds, sinks)
        doors = py - sinks
        bindings = lane_bindings(doors, hs)
        files = set(py)
        wide = []
        for b in sorted(bindings):
            closure, scope = binding_closure(b)
            if closure is None:
                wide.append(f"{b}: {scope}")
                files.add(os.path.join(BINDINGS, b + ".mojo"))
                continue
            if scope == "tree":
                wide.append(f"{b}: {scope}")
            files |= closure
        sources[lane] = files
        why[lane] = dict(python=len(py), bindings=sorted(bindings), mojo=len(files - py), wide=wide)
    _DERIVED = (sources, why)
    return _DERIVED


def reverse(sources):
    out = {}
    for lane, files in sources.items():
        for rel in files:
            out.setdefault(rel, set()).add(lane)
    return out


def identity_path(rel):
    """A file that can move a lane's bits: Mojo and Python sources. Build
    shells, data and prose are attributed by the selector's own rules."""
    return rel.endswith((".mojo", ".py"))


def compare(out=print, full=False):
    """Every disagreement between the graph map and `lane_select.lane_sources()`.

    NARROW-NOT-GRAPH is the list that matters: a file the selector attributes
    to a lane that no import path from that lane reaches. Each entry is either
    a declaration the manifest kept that the code no longer bears out, or an
    import form this walk does not read; both are defects to fix, never to
    file. GRAPH-NOT-NARROW is the expected direction (the selector's per-export
    narrowing) and is summarized per file. Returns the narrow-not-graph
    count."""
    graph, gwhy = derive()
    narrow, _ = lane_select.lane_sources()
    grev, nrev = reverse(graph), reverse(narrow)
    lanes = sorted(set(graph) | set(narrow))
    out(f"# lanes: graph {len(graph)}, selector {len(narrow)}; files: graph {len(grev)}, selector {len(nrev)}")
    wide = {lane: w for lane, w in gwhy.items() if w["wide"]}
    if wide:
        out(f"# {len(wide)} lane(s) carry a binding whose closure is the whole tree: "
            + "; ".join(f"{lane}: {','.join(w['wide'])}" for lane, w in sorted(wide.items())[:5]))
    only_narrow = {}
    for lane in lanes:
        for rel in sorted(narrow.get(lane, set()) - graph.get(lane, set())):
            if identity_path(rel):
                only_narrow.setdefault(rel, set()).add(lane)
    out(f"\n== NARROW-NOT-GRAPH: {len(only_narrow)} file(s) the selector attributes to a lane "
        "no import path from that lane reaches")
    for rel in sorted(only_narrow):
        ls = sorted(only_narrow[rel])
        out(f"  {rel}: {len(ls)} lane(s): {','.join(ls[:6])}{'...' if len(ls) > 6 else ''}")
    only_graph = {}
    for lane in lanes:
        for rel in sorted(graph.get(lane, set()) - narrow.get(lane, set())):
            if identity_path(rel):
                only_graph.setdefault(rel, set()).add(lane)
    out(f"\n== GRAPH-NOT-NARROW: {len(only_graph)} file(s) some lane reaches by import that the selector "
        "does not attribute to it (the per-export narrowing)")
    rows = sorted(only_graph.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for rel, ls in (rows if full else rows[:40]):
        out(f"  {rel}: graph {len(grev.get(rel, ()))} lane(s), selector {len(nrev.get(rel, ()))}, "
            f"+{len(ls)}")
    if not full and len(rows) > 40:
        out(f"  ... {len(rows) - 40} more (--full)")
    every = sorted(rel for rel, ls in grev.items() if len(ls) == len(graph) and identity_path(rel))
    out(f"\n== {len(every)} identity file(s) every lane reaches by import (selector: "
        f"{sum(1 for rel, ls in nrev.items() if len(ls) == len(narrow) and identity_path(rel))})")
    counts = sorted(len(ls) for rel, ls in grev.items() if identity_path(rel))
    ncounts = sorted(len(ls) for rel, ls in nrev.items() if identity_path(rel))
    out(f"== median lanes per identity file: graph {counts[len(counts) // 2]}, "
        f"selector {ncounts[len(ncounts) // 2]}")
    return len(only_narrow)


def check_stamps(out=print):
    """Every stamp under python/.binding-stamps/ that records its closure must
    record exactly the files this map derives for that binding's script."""
    import binding_stamps
    bad, seen = [], 0
    for sp in sorted(Path(binding_stamps.STAMPS).glob("*.json")) if Path(binding_stamps.STAMPS).is_dir() else []:
        rec = json.loads(sp.read_text())
        if "sources" not in rec:
            continue
        seen += 1
        derived, _ = mojo_closure(os.path.join(BINDINGS, rec["script"]))
        recorded = set(rec["sources"]) - {"pixi.toml", "pixi.lock"}
        if recorded != derived:
            bad.append(f"{rec['binding']}: stamp {len(recorded)} files, graph {len(derived)}; "
                       f"only in stamp {sorted(recorded - derived)[:4]}, only in graph {sorted(derived - recorded)[:4]}")
    for line in bad:
        out(line)
    out(f"# {seen} stamp(s) with a recorded closure, {len(bad)} disagree with the import graph"
        + ("" if seen else " (no built binding carries one; build first)"))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--compare", action="store_true", help="disagreements against lane_select's map")
    ap.add_argument("--full", action="store_true", help="with --compare, every graph-not-narrow file")
    ap.add_argument("--lane", metavar="NAME", help="print one lane's derived files")
    ap.add_argument("--file", metavar="PATH", help="print the lanes that reach one file by import")
    ap.add_argument("--json", metavar="PATH", help="write lane -> files as JSON")
    ap.add_argument("--check-stamps", action="store_true", help="recorded build closures equal the graph's")
    args = ap.parse_args(argv)
    if args.check_stamps:
        return check_stamps()
    if args.compare:
        n = compare(full=args.full)
        return 1 if n else 0
    sources, why = derive()
    if args.lane:
        if args.lane not in sources:
            raise SystemExit(f"REFUSING: no lane named {args.lane}")
        ev = why[args.lane]
        print(f"# {args.lane}: python {ev['python']} mojo {ev['mojo']} bindings {','.join(ev['bindings']) or '-'}"
              + (f" WIDE {ev['wide']}" if ev["wide"] else ""))
        for rel in sorted(sources[args.lane]):
            print(rel)
    if args.file:
        lanes = sorted(reverse(sources).get(args.file, ()))
        print(f"# {args.file}: {len(lanes)} lane(s)")
        print(",".join(lanes))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({lane: sorted(files) for lane, files in sources.items()}, fh, indent=1)
    if not (args.lane or args.file or args.json):
        rev = reverse(sources)
        print(f"# {len(sources)} lanes, {len(rev)} files; --compare, --lane, --file, --json, --check-stamps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
