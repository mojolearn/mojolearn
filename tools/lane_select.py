#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHICH LANES CAN A CHANGE POSSIBLY MOVE (lane/lane-selector, 2026-09-16).

Checking a two-file change used to mean running a sweep of identity lanes,
and on the Mac that sweep is Metal, one job at a time, hours of the scarcest
resource we own. Nothing in the tree could answer "run only the lanes this
change can affect", so the answer was always "run everything".

This file answers it. For every lane in `tools/identity_break.py`'s registry
it derives the source files the lane actually exercises, inverts that into
file -> lanes, and selects the lanes a set of changed paths can reach.

DERIVED, NOT HAND-WRITTEN. A hand-kept list rots silently and then answers
"nothing is affected" long after that stopped being true, which is the exact
shape of six failures in this repository. Every edge here comes from a
declaration that something else already enforces:

  the registry      `identity_break.LANES` itself, read by IMPORT, not by
                    grep. `--count` prints how many; on 2026-09-16 that was
                    211 against the 188 `grep -c '@lane('` finds, because 23
                    lanes register by call rather than by decorator (the kde,
                    knn, radius, gp and gmm families). The total is never
                    written down, here or in the docs: four of them were in
                    circulation in one afternoon.
  the lane body     the lane function's own code object: the names it touches
                    (`ml.RandomForestClassifier`, `ml.linalg`, the module's
                    own helpers, transitively) are in `co_names`.
  the Python door   `python/mojolearn/*.py`, indexed by the classes and
                    functions each file defines, then closed over the
                    package's own imports.
  the binding       the `_mojolearn_*` names those Python files name, which
                    are `bindings/_mojolearn_*.mojo`.
  the Mojo tree     each binding's own `from a.b.c import ...` lines, resolved
                    to `a/b/c.mojo` and followed transitively.
  the CPU surface   `python/mojolearn/host_surface.py`: a family declares the
                    lanes it covers, and the family's host binding is another
                    Mojo seed for those lanes.

CONSERVATIVE BY CONSTRUCTION. A selector that misses an affected lane is far
worse than one that runs a few extra, so:

  * a changed path this map does not attribute selects EVERY lane, and the
    reason is printed, never swallowed;
  * `tools/identity_break.py` selects every lane unless the diff touches
    ONLY lane function bodies, in which case it selects exactly those lanes
    (`harness_lanes` below, which refuses to narrow when a shared helper,
    a fixture, a constant or a registration loop moved);
  * only doc and evidence paths are treated as inert, by an explicit list.

WHAT IT REFUSES TO DO. It never returns an empty selection quietly. An empty
answer for a non-empty diff is reported as UNATTRIBUTED and reads as "run
everything", because a selector that silently selects nothing makes every
change look verified while checking nothing.

    python3 tools/lane_select.py --changed-since origin/main
    python3 tools/lane_select.py --lanes-for-paths glm/host/qn_oracle.mojo
    python3 tools/lane_select.py --lane logistic --why
    python3 tools/lane_select.py --all --shards 8
    python3 tools/lane_select.py --selfcheck

`tools/verify_lanes.py` is the one command that RUNS what this selects, for
one lane, for a change, or for everything. The tiers are
docs/lanes/VERIFICATION_TIERS.md.
"""
import argparse
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join("python", "mojolearn")
HARNESS = os.path.join("tools", "identity_break.py")
MANIFEST = os.path.join(PKG, "host_surface.py")

#: Paths that cannot move a bit. Prose, evidence, and the workflow files that
#: schedule runs rather than compute anything. EVERYTHING ELSE that this map
#: cannot attribute selects every lane, so this list is the only place where
#: "no lanes" is a legitimate answer, and it is short on purpose.
INERT_SUFFIXES = (".md", ".txt", ".tsv", ".cff", ".log", ".patch")
INERT_PREFIXES = (
    "docs/", "archive/", "parked/", "bench/results/", "catboost_info/",
    "run/", "build/", ".github/", "AUTHORS", "CHANGELOG", "CITATION",
    "CONTRIBUTING", "GOVERNANCE", "LICENSE", "NOTICE", "README", "ROADMAP",
    "SECURITY", "SUPPORT",
)

#: Paths whose change reaches every lane at once. The harness is here with a
#: refinement (`harness_lanes`); the manifest is here without one, because it
#: decides which lanes have a CPU route at all.
GLOBAL_PATHS = (HARNESS, MANIFEST)

_BINDING_RE = re.compile(r"_mojolearn[a-z0-9_]*")
_MOJO_IMPORT_RE = re.compile(r"^\s*(?:from\s+([a-zA-Z0-9_.]+)\s+import|import\s+([a-zA-Z0-9_.]+))")

#: A Python file that names more than this many bindings is a REGISTRY, not a
#: lane's door: `_backend.py` lists every binding it can load, `host_surface.py`
#: every host family, `_classical_host.py` every classical route, `__init__.py`
#: every public import. Measured on main 2026-09-16: 55, 43, 25 and 8 bindings
#: against 1 to 3 for a real estimator module.
#:
#: Harvesting binding names from those files gave EVERY lane EVERY binding and
#: so made every change select every lane, which is a selector that cannot
#: narrow anything. They are therefore sinks: their binding names are not
#: evidence for any one lane and the import walk stops at them. The
#: conservatism that buys is paid back exactly where it was taken, in
#: `select`: a change TO one of these files selects every lane, because the
#: edges it would have contributed were the ones dropped here.
ENUMERATOR_MAX_BINDINGS = 3


#: PER-FILE RESULTS ARE MEMOIZED, because the map asks the same question of
#: the same file once per lane. Without this, deriving the map parsed each
#: Python door three times for every one of the lanes that reach it: measured
#: 2026-09-16 on this tree, 146 s for one `lane_sources()` and 24 minutes for
#: the property tests, which is a poor look on the tool whose whole purpose is
#: verification TIME. Every cache below is keyed by path and holds a pure
#: function of that file's CONTENT; the tree does not change inside one run,
#: and nothing here writes a file. Same answers, 26x less of them.
_CACHES = []


def _by_path(fn):
    cache = {}
    _CACHES.append(cache)

    def wrapped(path):
        if path not in cache:
            cache[path] = fn(path)
        return cache[path]

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    wrapped.cache = cache          # tests seed this to ask about a file that is not in the tree
    return wrapped


def reset_caches():
    """Drop every memo. Only a caller that edits the tree mid-process needs
    this; no code path in this repository does."""
    global _ENUMERATORS, _LANE_SOURCES
    for cache in _CACHES:
        cache.clear()
    _ENUMERATORS = None
    _LANE_SOURCES = None


@_by_path
def _read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8", errors="replace") as fh:
        return fh.read()


@_by_path
def _parse(path):
    """One file's AST, or None when it is unreadable or not Python."""
    try:
        return ast.parse(_read(path))
    except (OSError, SyntaxError):
        return None


_MODULES = {}


def _load_module(name, path):
    if name not in _MODULES:
        full = os.path.join(ROOT, path)
        spec = importlib.util.spec_from_file_location(name, full)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _MODULES[name] = mod
    return _MODULES[name]


def identity_break():
    """The harness module, IMPORTED. `LANES` is the registry; reading it any
    other way (a grep over decorators) undercounts the lanes that register by
    call and has already produced four different lane totals in one day."""
    return _load_module("mojolearn_identity_break", HARNESS)


def host_surface():
    return _load_module("mojolearn_host_surface", MANIFEST)


def all_lanes():
    """Every registered lane, in registry order. THE lane set; `--count`
    prints its size and nothing in this repository writes that number down."""
    return list(identity_break().LANES)


# ------------------------------------------------------------------ the map

def _code_names(fn, module_globals, seen=None):
    """Every name a lane function touches, following the module's own helper
    functions transitively. `ml.RandomForestClassifier(...)` leaves
    'RandomForestClassifier' in `co_names`; `_km_probe(...)` leaves the
    helper's name, and the helper's own names come back with it."""
    if seen is None:
        seen = set()
    out = set()
    stack = [fn.__code__]
    if getattr(fn, "__closure__", None):
        for cell in fn.__closure__:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if callable(val) and hasattr(val, "__code__") and id(val) not in seen:
                seen.add(id(val))
                out |= _code_names(val, module_globals, seen)
    while stack:
        code = stack.pop()
        out |= set(code.co_names)
        for const in code.co_consts:
            if hasattr(const, "co_names"):
                stack.append(const)
    for name in list(out):
        helper = module_globals.get(name)
        if callable(helper) and hasattr(helper, "__code__") and id(helper) not in seen:
            seen.add(id(helper))
            out |= _code_names(helper, module_globals, seen)
    return out


def _python_files():
    """Every tracked Python file of the package, excluding its test modules
    (a test cannot change a lane's bits)."""
    out = []
    base = os.path.join(ROOT, PKG)
    for name in sorted(os.listdir(base)):
        if name.endswith(".py"):
            out.append(os.path.join(PKG, name))
    return out


def _python_symbols(files):
    """symbol -> files that define it: every top-level class and def, plus
    each file's own module name, so `ml.linalg` and `ml.metrics` resolve."""
    index = {}
    for rel in files:
        tree = _parse(rel)
        if tree is None:
            continue
        index.setdefault(os.path.basename(rel)[:-3].lstrip("_"), set()).add(rel)
        index.setdefault(os.path.basename(rel)[:-3], set()).add(rel)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                index.setdefault(node.name, set()).add(rel)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        index.setdefault(target.id, set()).add(rel)
    return index


@_by_path
def _python_imports(rel):
    """The package's own imports of one file, as file paths."""
    out = set()
    tree = _parse(rel)
    if tree is None:
        return out
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.module:
                names.add(node.module.split(".")[0])
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mojolearn."):
                    names.add(alias.name.split(".")[1])
    for name in names:
        cand = os.path.join(PKG, name + ".py")
        if os.path.exists(os.path.join(ROOT, cand)):
            out.add(cand)
    return out


_ENUMERATORS = None


def enumerator_files():
    """The package's registry files, found by counting rather than named by
    hand: a file that names more than ENUMERATOR_MAX_BINDINGS real bindings
    lists the whole surface and says nothing about one lane. Recomputed from
    the tree on every run, so a new registry file is caught the day it lands
    instead of quietly widening every selection."""
    global _ENUMERATORS
    if _ENUMERATORS is None:
        out = set()
        for rel in _python_files():
            named = {b for b in set(_BINDING_RE.findall(_read(rel)))
                     if os.path.exists(os.path.join(ROOT, "bindings", b + ".mojo"))}
            if len(named) > ENUMERATOR_MAX_BINDINGS:
                out.add(rel)
        _ENUMERATORS = out
    return _ENUMERATORS


def _python_closure(seeds, sinks=()):
    """The package's own import closure of `seeds`. A sink is included but not
    followed: the walk stops at a registry rather than stepping through it into
    every other family."""
    out, stack = set(), list(seeds)
    while stack:
        rel = stack.pop()
        if rel in out:
            continue
        out.add(rel)
        if rel not in sinks:
            stack.extend(_python_imports(rel))
    return out


@_by_path
def _mojo_imports(rel):
    """The repository's own Mojo imports of one file. `from cluster.impl.kmeans
    import x` resolves to cluster/impl/kmeans.mojo; `from max.gpu.host import
    ...` resolves to nothing here and is dropped, which is what makes the
    toolchain's own modules fall out."""
    out = set()
    try:
        text = _read(rel)
    except OSError:
        return out
    for line in text.splitlines():
        m = _MOJO_IMPORT_RE.match(line)
        if not m:
            continue
        dotted = m.group(1) or m.group(2)
        parts = dotted.split(".")
        for cand in (os.path.join(*parts) + ".mojo", os.path.join(*parts, "__init__.mojo")):
            if os.path.exists(os.path.join(ROOT, cand)):
                out.add(cand)
    return out


def _mojo_closure(seeds):
    out, stack = set(), [s for s in seeds if os.path.exists(os.path.join(ROOT, s))]
    while stack:
        rel = stack.pop()
        if rel in out:
            continue
        out.add(rel)
        stack.extend(_mojo_imports(rel))
    return out


def _mojo_blocks(text):
    """A Mojo source's top-level `fn` and `def` blocks, name -> body text."""
    lines = text.splitlines()
    starts = []
    for i, line in enumerate(lines):
        m = re.match(r"^(?:fn|def)\s+([A-Za-z0-9_]+)", line)
        if m:
            starts.append((i, m.group(1)))
    out = {}
    for k, (i, name) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        out[name] = "\n".join(lines[i:end])
    return out


@_by_path
def _mojo_blocks_for(rel):
    """`_mojo_blocks` of one file, memoized: a binding is asked for its blocks
    once per distinct set of exports a lane's door calls."""
    return _mojo_blocks(_read(rel))


@_by_path
def _mojo_import_symbols(rel):
    """symbol -> the file it is imported from, for one Mojo source, including
    the parenthesized multi-line form the bindings use."""
    try:
        text = _read(rel)
    except OSError:
        return {}
    text = re.sub(r"\(\s*([^()]*?)\s*\)", lambda m: m.group(1).replace("\n", " "), text, flags=re.S)
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*from\s+([A-Za-z0-9_.]+)\s+import\s+(.+)$", line)
        if not m:
            continue
        parts = m.group(1).split(".")
        files = {c for c in (os.path.join(*parts) + ".mojo", os.path.join(*parts, "__init__.mojo"))
                 if os.path.exists(os.path.join(ROOT, c))}
        if not files:
            continue                         # the toolchain's own modules, not ours
        for name in m.group(2).split(","):
            name = name.strip().split(" as ")[0].strip()
            if name and name != "*":
                out.setdefault(name, set()).update(files)
    return out


@_by_path
def _binding_exports(rel):
    """export name -> the binding function implementing it, from the
    `module.def_function[impl]("name")` registrations a binding ends with."""
    try:
        text = _read(rel)
    except OSError:
        return {}
    return {m.group(2): m.group(1) for m in
            re.finditer(r"def_function\[\s*([A-Za-z0-9_]+)\s*\]\s*\(\s*\"([A-Za-z0-9_]+)\"", text)}


#: Calls that RESOLVE a binding by name, and the assignment targets that hold
#: one. `_backend.binding("_mojolearn_rf")`, `self._bind("_mojolearn")`,
#: `load_host_module(basename)`, `_BINDING = "_mojolearn_rf"`.
_BINDING_CALLS = frozenset({"binding", "_bind", "load_host_module", "host_module_path"})
_BINDING_TARGETS = ("_BINDING", "_EXT_NAME", "_EXT", "_MODULE_NAME", "_EXTENSION")


@_by_path
def _binding_names(rel):
    """The bindings a Python file actually RESOLVES, read from its syntax.

    NOT a text search. `_BINDING_RE.findall` over the file matches binding
    names in DOCSTRINGS AND COMMENTS, and this codebase explains itself at
    length: on 2026-09-16 that gave every lane `_mojolearn_forest_host` and
    `_mojolearn_byte_lm_host` because some shared door mentions them in prose,
    every lane then hit those bindings' whole-closure fallback, and
    core/gbdt_host_predict.mojo selected every lane at once. A sentence about a
    binding is not a call into it."""
    out = set()
    tree = _parse(rel)
    if tree is None:
        return out

    def add(value):
        if isinstance(value, str) and value.startswith("_mojolearn"):
            out.add(value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in _BINDING_CALLS:
                for arg in node.args:
                    if isinstance(arg, ast.Constant):
                        add(arg.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                label = getattr(target, "id", None) or getattr(target, "attr", None)
                if label and label.upper().endswith(_BINDING_TARGETS):
                    if isinstance(node.value, ast.Constant):
                        add(node.value.value)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                add(alias.name)
    return out


@_by_path
def _file_called_names(rel):
    """One Python file's attribute names and identifier-shaped string
    constants: `self._bind("_mojolearn").kmeans_fit(` leaves 'kmeans_fit',
    and `getattr(binding, "qn_fit")` leaves 'qn_fit'."""
    out = set()
    tree = _parse(rel)
    if tree is None:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.isidentifier():
            out.add(node.value)
    return out


def _called_names(files):
    """The same, over a set of files. Intersected with a binding's exports,
    this is which entry points a lane's door actually calls."""
    out = set()
    for rel in files:
        out |= _file_called_names(rel)
    return out


_LANE_SOURCES = None


def lane_sources():
    """lane -> the source files it exercises, derived. Also returns the
    per-lane evidence (`why`) so a selection can say what it rested on.

    MEMOIZED for the life of the process. `select()` and the property tests
    each ask for the map several times, and deriving it is the expensive part
    of every command here."""
    global _LANE_SOURCES
    if _LANE_SOURCES is not None:
        return _LANE_SOURCES
    ib = identity_break()
    hs = host_surface()
    py_files = _python_files()
    symbols = _python_symbols(py_files)
    routed = hs.routed_modules()                 # GPU binding -> host binding
    adapted = {k: v["family"] for k, v in hs.ADAPTED_MODULES.items()}

    # family -> the lanes it declares. A family's own files are resolved PER
    # LANE below, through the exports that lane's door actually calls.
    family_lanes = {f["family"]: set(f["training_lanes"]) | set(f["inference_lanes"])
                    for f in hs.FAMILIES}
    family_by_name = {f["family"]: f for f in hs.FAMILIES}
    family_of_binding = {f["binding"]: f["family"] for f in hs.FAMILIES}
    seed_cache = {}

    def binding_seeds(src, used, wide=True):
        """The Mojo files ONE binding contributes to a lane whose door calls
        `used`.

        A binding is a multiplexer: bindings/_mojolearn_core_host.mojo carries
        the k-means, DBSCAN, k-NN and scaler oracles at once, so following its
        whole import closure handed every lane of the family every oracle in
        it, and a change to one oracle selected the whole registry (measured
        2026-09-16, when it held 199: glm/host/qn_oracle.mojo selected 163 and
        cluster/host/kmeans_oracle.mojo all 199). Each EXPORT is resolved to
        the imported symbols ITS OWN function body names, and only those
        modules are followed. The binding source itself is always included,
        because a change to it does reach the lane, but its imports are not
        followed wholesale.

        A lane whose door calls none of this binding's exports gets the whole
        closure back. "I could not tell" has to stay wide.

        EXCEPT FOR A BINDING EVERY LANE REACHES (`wide` False). Three doors
        are in EVERY lane's import closure on this tree, through the shared
        buffer helpers: `_forest_host.py`, `_byte_lm_impl.py` and
        `_byte_lm_host.py`. So every lane resolved the forest and byte LM
        bindings, every lane's door called none of THEIR exports, every lane
        took this whole-closure fallback, and `core/gbdt_host_predict.mojo`
        and the whole mamba tree selected all 211 lanes. That is the same
        symptom the syntax rule above was written for, arriving by a second
        road: the prose was one source of the edge and the import closure is
        another. Reaching a binding that everything reaches says nothing about
        one lane, so such a binding contributes its SOURCE only, and a change
        to the binding file still selects every lane."""
        if wide is not True:
            # A BINDING EVERY LANE REACHES. `wide` is False for a lane the CPU
            # manifest does not name for this family (source only, so a change
            # to the binding file still selects the lane) and "whole" for one
            # it does name, which takes the binding entire because the manifest
            # is the declaration that this lane belongs to it.
            return _mojo_closure([src]) if wide == "whole" else {src}
        exports = _binding_exports(src)
        hit = tuple(sorted(e for e in used if e in exports))
        key = (src, hit)
        if key not in seed_cache:
            if not hit:
                seed_cache[key] = _mojo_closure([src])
            else:
                blocks = _mojo_blocks_for(src)
                syms = _mojo_import_symbols(src)
                seeds = set()
                for export in hit:
                    body = blocks.get(exports[export], "")
                    for sym, files in syms.items():
                        if re.search(r"\b%s\b" % re.escape(sym), body):
                            seeds |= files
                seed_cache[key] = _mojo_closure(seeds) | {src}
        return seed_cache[key]

    sources, why = {}, {}
    sinks = enumerator_files()

    # PASS ONE: each lane's Python doors and the bindings they resolve. Held
    # first because the next pass needs to know which bindings EVERY lane
    # resolves, and that is a measurement over all the lanes, not a list.
    reach = {}
    for lane, fn in ib.LANES.items():
        names = _code_names(fn, vars(ib))
        seeds, matched = set(), set()
        for name in names:
            for rel in symbols.get(name, ()):
                seeds.add(rel)
                matched.add(name)
        files = _python_closure(seeds, sinks)
        doors = files - set(sinks)
        bindings = set()
        for rel in doors:
            bindings |= _binding_names(rel)          # resolved by syntax, never by prose
        bindings = {b for b in bindings
                    if os.path.exists(os.path.join(ROOT, "bindings", b + ".mojo"))}
        reach[lane] = (matched, files, doors, bindings)
    ubiquitous = set.intersection(*[b for _, _, _, b in reach.values()]) if reach else set()

    def _declared(lane, binding):
        """Does the CPU manifest name this lane for the family that owns this
        binding? That is the one declaration in the tree which says a lane
        really belongs to a family, and it is what keeps the byte LM lanes
        wide while `ols` is not."""
        for b in (binding, routed.get(binding)):
            fam = family_of_binding.get(b)
            if fam and lane in family_lanes.get(fam, set()):
                return True
        return binding in adapted and lane in family_lanes.get(adapted[binding], set())

    for lane in reach:
        matched, files, doors, bindings = reach[lane]
        used = _called_names(doors)
        mojo = set()
        for binding in bindings:
            gpu_src = os.path.join("bindings", binding + ".mojo")
            wide = True if binding not in ubiquitous else \
                ("whole" if _declared(lane, binding) else False)
            mojo |= binding_seeds(gpu_src, used, wide)
            host = routed.get(binding)
            if host:
                host_src = os.path.join("bindings", host + ".mojo")
                # WHOSE CPU PATH IS THIS. The manifest DECLARES which lanes a
                # host family serves, so a lane it does not name is not served
                # by this binding and gets only the binding source (a change to
                # the file still selects the lane) rather than its whole oracle
                # closure. Without this rule every lane that merely referenced
                # the GPU family inherited every oracle in its host binding:
                # glm/host/qn_oracle.mojo selected 141 lanes on 2026-09-16,
                # among them agglomerative, dbscan and the gbdt lanes, none of
                # which has a quasi-Newton solver anywhere near it.
                fam = family_of_binding.get(host)
                if fam and lane in family_lanes.get(fam, set()):
                    # PER EXPORT, never "whole". A host binding is the
                    # multiplexer this rule was written for: taking
                    # _mojolearn_core_host entire hands every core lane the
                    # k-means, DBSCAN, k-NN and scaler oracles at once.
                    mojo |= binding_seeds(host_src, used)
                else:
                    mojo.add(host_src)
        fams = {f for f, lanes in family_lanes.items() if lane in lanes}
        fams |= {adapted[b] for b in bindings if b in adapted}
        extra = set()
        for fam in fams:
            spec = family_by_name[fam]
            src = hs.binding_source(fam)
            mojo |= binding_seeds(src, used)
            mojo.add(hs.build_shim(fam))
            if not any(e in used for e in _binding_exports(src)):
                # nothing resolved by name: fall back to what the manifest declares
                mojo |= set(spec.get("host_modules", ()))
            for door in (spec.get("loaded_by") or "").split(","):
                door = door.strip()
                if door and os.path.exists(os.path.join(ROOT, door)):
                    extra.add(door)
        sources[lane] = files | extra | mojo
        why[lane] = dict(symbols=sorted(matched), python=len(files), bindings=sorted(bindings),
                         families=sorted(fams), mojo=len(mojo), exports=len(used),
                         ubiquitous=sorted(ubiquitous))
    _LANE_SOURCES = (sources, why)
    return _LANE_SOURCES


def reverse_map(sources=None):
    """file -> the lanes that exercise it."""
    if sources is None:
        sources, _ = lane_sources()
    out = {}
    for lane, files in sources.items():
        for rel in files:
            out.setdefault(rel, set()).add(lane)
    return out


# ------------------------------------------------------------- the selector

def _is_inert(path):
    if path.endswith(INERT_SUFFIXES):
        return True
    return path.startswith(INERT_PREFIXES)


def harness_lanes(ref, path=HARNESS):
    """The lanes a diff of `tools/identity_break.py` can reach, or None when
    it can reach all of them.

    A lane commit usually edits ONE lane body in this file. Taking that at
    face value would be a guess, so this compares the top-level statements of
    the two revisions: a changed statement that IS a `@lane`-decorated
    function attributes to that lane, and ANY other changed top-level
    statement (a helper, a fixture, a constant, one of the registration
    loops that build the kde, knn, radius, gp and gmm lanes) returns None,
    which the caller reads as every lane."""
    try:
        old = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:{path}"],
                             capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
    try:
        new = _read(path)
        old_tree, new_tree = ast.parse(old), ast.parse(new)
    except (OSError, SyntaxError):
        return None

    def segments(tree, text):
        out = []
        lines = text.splitlines()
        for node in tree.body:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            body = "\n".join(lines[start - 1:node.end_lineno])
            lane = None
            for dec in getattr(node, "decorator_list", []):
                if (isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "lane"
                        and dec.args and isinstance(dec.args[0], ast.Constant)):
                    lane = dec.args[0].value
            key = lane or (getattr(node, "name", None) or f"stmt@{start}:{ast.dump(node)[:80]}")
            out.append((key, lane, body))
        return out

    old_seg = {k: (lane, body) for k, lane, body in segments(old_tree, old)}
    new_seg = {k: (lane, body) for k, lane, body in segments(new_tree, new)}
    touched = set()
    for key in set(old_seg) | set(new_seg):
        o, n = old_seg.get(key), new_seg.get(key)
        if o == n:
            continue
        lane = (n or o)[0]
        if lane is None:
            return None                     # a shared statement moved: every lane
        touched.add(lane)
    return sorted(touched)


def changed_paths(ref):
    """Paths that differ from `ref`, including uncommitted work: a change you
    have not committed is still a change this run has to cover."""
    out = set()
    for args in (["diff", "--name-only", f"{ref}...HEAD"], ["diff", "--name-only", "HEAD"],
                 ["ls-files", "--others", "--exclude-standard"]):
        try:
            res = subprocess.run(["git", "-C", ROOT] + args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            continue
        out |= {line.strip() for line in res.stdout.splitlines() if line.strip()}
    return sorted(out)


def select(paths, ref=None, sources=None):
    """The lanes `paths` can affect.

    Returns a dict: `lanes` (sorted), `fallback` (True when the answer is
    every lane), `reasons` (path -> why it selected what it did) and
    `unattributed` (the paths that forced the fallback). The caller prints
    every one of these; a fallback that is not said out loud is a selector
    that lies."""
    if sources is None:
        sources, _ = lane_sources()
    every = sorted(sources)
    rev = reverse_map(sources)
    lanes, reasons, unattributed, inert = set(), {}, [], []
    fallback = False
    for path in paths:
        path = path.strip()
        if not path:
            continue
        if re.search(r"\s", path):
            # SEVERAL PATHS IN ONE ARGUMENT. zsh does not word-split an
            # unquoted variable, so `--lanes-for-paths $files` arrives as a
            # single string. On 2026-09-16 that string began with CHANGELOG.md,
            # matched the inert prefix rule, and a twelve-file commit selected
            # ZERO lanes without a word of complaint. An argument that is not
            # one path is never inert.
            fallback = True
            unattributed.append(path)
            reasons[path] = ("NOT A SINGLE PATH (whitespace inside it, so this is several paths "
                             "in one argument; quote it or pass them separately): every lane")
            continue
        if _is_inert(path):
            inert.append(path)
            reasons[path] = "inert (prose or evidence)"
            continue
        if path == HARNESS:
            touched = harness_lanes(ref) if ref else None
            if touched is None:
                fallback = True
                unattributed.append(path)
                reasons[path] = ("the harness itself changed outside a lane body "
                                 "(or no ref to compare against): every lane")
            else:
                lanes |= set(touched)
                reasons[path] = f"harness diff touches only these lane bodies: {','.join(touched) or 'none'}"
            continue
        if path in GLOBAL_PATHS:
            fallback = True
            unattributed.append(path)
            reasons[path] = "declares the CPU surface itself: every lane"
            continue
        if path in enumerator_files():
            fallback = True
            unattributed.append(path)
            reasons[path] = ("a registry of the whole binding surface, so the map drops its "
                             "per-lane edges on purpose: every lane")
            continue
        hit = rev.get(path)
        if hit:
            lanes |= hit
            reasons[path] = f"{len(hit)} lane(s)"
            continue
        fallback = True
        unattributed.append(path)
        reasons[path] = "NOT ATTRIBUTABLE: no lane's derived source set names it, so every lane"
    if fallback:
        lanes = set(every)
    return dict(lanes=sorted(lanes, key=every.index), fallback=fallback, reasons=reasons,
                unattributed=sorted(unattributed), inert=sorted(inert), total=len(every))


# --------------------------------------------------------------- sharding

def shard(lanes, shards):
    """`lanes` split into at most `shards` deterministic lists. The split is
    tools/cpu_identity_gate_check.py's, so the two agree on balance and on
    order, and the same lane set always produces the same shards. The union
    is asserted here and again at merge time."""
    path = os.path.join(ROOT, "tools", "cpu_identity_gate_check.py")
    spec = importlib.util.spec_from_file_location("mojolearn_cpu_gate_check", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    out, load = mod.shard_lanes(list(lanes), shards)
    union = sorted(sum(out, []))
    if union != sorted(lanes):
        raise SystemExit(f"REFUSING: the shards are not the lane set ({len(union)} vs {len(set(lanes))})")
    return out, load


# ------------------------------------------------------------------- CLI

def selfcheck():
    """Every lane maps to at least one Python file and one Mojo file. A lane
    whose map is empty would select nothing for a change to its own code,
    which is the silent-pass this whole file exists to prevent."""
    sources, why = lane_sources()
    bad = []
    for lane in sorted(sources):
        files = sources[lane]
        mojo = [f for f in files if f.endswith(".mojo")]
        py = [f for f in files if f.endswith(".py")]
        if not mojo or not py:
            bad.append((lane, len(py), len(mojo), why[lane]))
    print(f"lanes {len(sources)}")
    print(f"files mapped {len(reverse_map(sources))}")
    for rel in sorted(enumerator_files()):
        named = len({b for b in set(_BINDING_RE.findall(_read(rel)))
                     if os.path.exists(os.path.join(ROOT, "bindings", b + ".mojo"))})
        print(f"registry {rel}: names {named} bindings, so a change to it selects every lane")
    for lane, npy, nmojo, ev in bad:
        print(f"UNMAPPED {lane}: {npy} python, {nmojo} mojo, symbols={ev['symbols'][:6]}")
    print(f"selfcheck {'OK' if not bad else 'FAIL'}: {len(bad)} lane(s) with an empty side")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--changed-since", metavar="REF",
                    help="select the lanes the diff against REF can affect (origin/main is the usual one)")
    ap.add_argument("--lanes-for-paths", nargs="+", metavar="PATH",
                    help="select the lanes these paths can affect")
    ap.add_argument("--lane", action="append", default=[], metavar="NAME",
                    help="name a lane explicitly; repeatable")
    ap.add_argument("--all", action="store_true", help="every registered lane")
    ap.add_argument("--why", action="store_true", help="print what each selected lane rested on")
    ap.add_argument("--files", action="store_true", help="print the selected lanes' source files")
    ap.add_argument("--shards", type=int, default=0, metavar="N", help="also print a deterministic N-way split")
    ap.add_argument("--json", default="", metavar="PATH", help="write the selection as JSON")
    ap.add_argument("--selfcheck", action="store_true", help="every lane maps to real files")
    ap.add_argument("--count", action="store_true", help="print the registry's lane count and exit")
    args = ap.parse_args(argv)

    if args.count:
        print(len(all_lanes()))
        return 0
    if args.selfcheck:
        return selfcheck()

    sources, why = lane_sources()
    every = sorted(sources, key=list(sources).index)
    if args.all:
        sel = dict(lanes=list(sources), fallback=False, reasons={"--all": "asked for every lane"},
                   unattributed=[], inert=[], total=len(sources))
    elif args.lane:
        unknown = [n for n in args.lane if n not in sources]
        if unknown:
            raise SystemExit(f"REFUSING: --lane names no lane: {unknown}")
        sel = dict(lanes=[n for n in every if n in set(args.lane)], fallback=False,
                   reasons={n: "named" for n in args.lane}, unattributed=[], inert=[], total=len(sources))
    elif args.lanes_for_paths:
        sel = select(args.lanes_for_paths, sources=sources)
    elif args.changed_since:
        paths = changed_paths(args.changed_since)
        sel = select(paths, ref=args.changed_since, sources=sources)
        sel["changed"] = paths
        print(f"# {len(paths)} changed path(s) against {args.changed_since}")
    else:
        ap.error("one of --all, --lane, --lanes-for-paths, --changed-since, --selfcheck, --count")

    for path, reason in sorted(sel["reasons"].items()):
        print(f"# {path}: {reason}")
    if sel["fallback"]:
        print("# FALLING BACK TO EVERY LANE. The blast radius of the paths above could not be "
              "determined, so this selection is not narrowed. Fix the map or say why, but do "
              "not read this as a narrow run.")
    print(f"# {len(sel['lanes'])} of {sel['total']} lanes selected")
    if args.why:
        for lane in sel["lanes"]:
            ev = why[lane]
            print(f"#   {lane}: bindings={','.join(ev['bindings']) or '-'} "
                  f"families={','.join(ev['families']) or '-'} "
                  f"python={ev['python']} mojo={ev['mojo']}")
    if args.files:
        files = sorted(set().union(*[sources[n] for n in sel["lanes"]])) if sel["lanes"] else []
        for rel in files:
            print(f"#   file {rel}")
    if args.shards:
        groups, load = shard(sel["lanes"], args.shards)
        for k, group in enumerate(groups):
            print(f"# shard {k} weight {load[k]} s: {','.join(group)}")
    print(",".join(sel["lanes"]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({k: v for k, v in sel.items() if k != "reasons"} | dict(reasons=sel["reasons"]),
                      fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
