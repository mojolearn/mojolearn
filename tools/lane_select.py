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
    Everything else that selects fewer than every lane rests on a DERIVED
    rule with its own docstring and its own test that the rule still widens
    where it must (2026-09-21, after the 0.8.12 Apple pass fell back on 148
    paths): `build_script_roots` (a build script selects the lanes of the
    binding it compiles), `pixi_tasks_only`, `NATIVE_INPUTS`,
    `_outside_python_unreachable`, `_unimported_mojo_unreachable`,
    `_package_file_unreachable`, `_lane_keyed_prose` and the corpus-only
    reading of `test_module_inert`.

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

#: THE SELECTION MACHINERY ITSELF. These files decide WHICH lanes run; they
#: cannot change what any lane computes, because nothing the harness or the
#: package imports reaches them (`test_the_selector_is_not_imported_by_what_it
#: _selects` holds that). Without this the lane that introduced them ran into
#: its own tool: on 2026-09-16 `--changed-since origin/main` on this branch
#: read all five as NOT ATTRIBUTABLE and fell back to all 211 lanes, which is
#: the full sweep this file exists to avoid. They are not folded into
#: INERT_PREFIXES because "inert (prose or evidence)" would be a false
#: description of a tool, and the reason printed should say what they are.
SELECTION_MACHINERY = (
    os.path.join("tools", "lane_select.py"),
    os.path.join("tools", "verify_lanes.py"),
    os.path.join("tools", "test_lane_select.py"),
    os.path.join("tools", "identity_iterate.py"),
    os.path.join("tools", "mac_slot.py"),
    os.path.join("tools", "test_algorithm_scope.py"),
    os.path.join("tools", "test_mac_slot.py"),
    os.path.join("tools", "test_identity_runtime.py"),
    os.path.join("tools", "check_python_gates.py"),
    os.path.join("tools", "check_python_gates.sh"),
    os.path.join("tools", "test_backend_control.py"),
    os.path.join("tools", "test_gate_scope.py"),
)

#: FILES THE HARNESS IMPORTS WHILE IT RECORDS A COLUMN. `lane_applicability.py`
#: was selection machinery until 2026-09-19, when `identity_break.py` began
#: importing it at run time to write `degenerate_lanes` into every column it
#: records. From then on a change to it changes what EVERY column says about
#: itself, and which lanes it names is decided by `degenerate()` over the
#: whole registry, so no one lane owns the effect: a change to one of these
#: selects every lane. They stay out of the reaching corpus like the machinery,
#: because their text names paths to READ them, not to run them in a lane.
HARNESS_RUNTIME_IMPORTS = (
    os.path.join("tools", "lane_applicability.py"),
)

#: Tools whose text is not evidence that a lane reaches a path.
_NOT_CORPUS = SELECTION_MACHINERY + HARNESS_RUNTIME_IMPORTS

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

#: A package file extended (subclassed or patched) by more than this many
#: others defines a base everything inherits, and "X extends it" then says
#: nothing about which lane X belongs to. Measured 2026-09-16: 23 for
#: `_mode.py` against 2 for the next, so the gap is not close.
UNIVERSAL_BASE_MAX_EXTENDERS = 3


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
    global _ENUMERATORS, _LANE_SOURCES, _SOURCE_HASHED, _TRACKED, _CONSTANTS, _EXTENDERS
    global _MOJO_CONFORMANCE, _MOJO_IMPORTERS, _PYTHON_FILES, _PKG_IMPORT_CLOSURE
    global _CORPUS, _CORPUS_TEXT, _REVERSE
    for cache in _CACHES:
        cache.clear()
    _GIT_SHOW.clear()
    _TRACKED = None
    _PYTHON_FILES = None
    _ENUMERATORS = None
    _LANE_SOURCES = None
    _SOURCE_HASHED = None
    _CONSTANTS = None
    _EXTENDERS = None
    _MOJO_CONFORMANCE = None
    _MOJO_IMPORTERS = None
    _PKG_IMPORT_CLOSURE = None
    _CORPUS = None
    _CORPUS_TEXT = None
    _REVERSE = None


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


_PYTHON_FILES = None


def _python_files():
    """Every tracked Python file of the package, excluding its test modules
    (a test cannot change a lane's bits).

    MEMOIZED: `_python_imports` asks for this listing once per package file to
    tell a module name from a class name, and the `os.listdir` behind it is
    the whole cost of that loop."""
    global _PYTHON_FILES
    if _PYTHON_FILES is not None:
        return _PYTHON_FILES
    out = []
    base = os.path.join(ROOT, PKG)
    for name in sorted(os.listdir(base)):
        if name.endswith(".py"):
            out.append(os.path.join(PKG, name))
    _PYTHON_FILES = out
    return out


def _public_rebindings(files):
    """name -> the package module `python/mojolearn/__init__.py` binds it FROM.

    A lane body writes `ml.UMAP`, and that attribute is bound by
    `__init__.py:147`, `from .umap import UMAP`, which executes
    python/mojolearn/umap.py, which is `from ._umap_impl import UMAP` and
    nothing else. Seeding only the file that DEFINES a class walks straight
    past that door: `umap.py`, `neural_network.py` and `language_model.py`
    were in no lane's map, and a change to one of them rebinds the public name
    every lane of that family calls.

    ONLY THROUGH `__init__.py`. Treating every `from .X import N` in the
    package as a binding of N was measured and is far too wide: it took the
    median file from 29 lanes to 60 and `neural_inference.py` from 21 to all
    212, because every impl module imports its neighbours. The package's own
    `__init__` is the one place that says which module the PUBLIC name comes
    from, which is the only rebinding a lane's `ml.<Name>` can go through."""
    out = {}
    listing = set(files)
    init = os.path.join(PKG, "__init__.py")
    tree = _parse(init)
    for node in (tree.body if tree is not None else ()):
        if not (isinstance(node, ast.ImportFrom) and node.level and node.module):
            continue
        rel = os.path.join(PKG, node.module.split(".")[0] + ".py")
        if rel not in listing or rel == init:
            continue
        for alias in node.names:
            if alias.name != "*":
                out.setdefault(alias.name, set()).add(rel)
    return out


def _python_symbols(files):
    """symbol -> files that define it: every top-level class and def, plus
    each file's own module name, so `ml.linalg` and `ml.metrics` resolve, plus
    the public door `__init__.py` binds the name from (`_public_rebindings`)."""
    index = {}
    for name, rels in _public_rebindings(files).items():
        index.setdefault(name, set()).update(rels)
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
    # AGAINST THE LISTING, NEVER `os.path.exists`. This asks whether a name a
    # file imports is itself a package module, and an imported name is often a
    # CLASS. On the macOS checkout the filesystem is case-insensitive, so
    # `from ._umap_impl import UMAP` answered yes to python/mojolearn/UMAP.py
    # and `from ._hdbscan_impl import HDBSCAN` to python/mojolearn/HDBSCAN.py.
    # Neither path is tracked. The map carried two files that exist only on
    # this laptop, HDBSCAN.py holding 24 lanes, while the real hdbscan.py and
    # umap.py, which are the public doors `__init__.py` binds those names
    # from, were in no lane's map at all. On the Linux boxes that run the CPU
    # column the same map is a different map.
    listing = set(_python_files())
    for name in names:
        cand = os.path.join(PKG, name + ".py")
        if cand in listing:
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


@_by_path
def _extends(rel):
    """The names this file's top-level classes INHERIT from, and the names it
    assigns an attribute on at module level.

    Both are edges that run BACKWARDS along the import graph, and the map
    followed imports forward only. `neural_inference.py` defines
    `Mamba1BlockInference(_RecurrentBlockInference, Mamba1Block)`: it imports
    `_mamba_impl`, so `_mamba_impl` never reaches IT, and the mamba and samba
    lanes were credited with three of the six lanes that file serves. Deleting
    those wrappers' `forward` overrides is about as load-bearing as an edit to
    that file gets, and the map called it none of their business.

    Reported by lane/stateful-cpu-decoding 2026-09-16. An under-attribution is
    silent and reads as a narrow PASS, which is the failure every other rule
    here is written against."""
    tree = _parse(rel)
    out = set()
    for node in (tree.body if tree else []):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = getattr(base, "id", None) or getattr(base, "attr", None)
                if name:
                    out.add(name)
        elif isinstance(node, ast.Assign):
            # MONKEYPATCHING. `SomeClass.method = f` at module level rebinds
            # behaviour on a class defined somewhere else entirely.
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    owner = getattr(target.value, "id", None)
                    if owner:
                        out.add(owner)
    return out


_EXTENDERS = None


def extenders():
    """file -> the package files it extends, by subclassing or by patching a
    class on. Built once over the package, then used to walk the import graph
    BACKWARDS from a lane's closure."""
    global _EXTENDERS
    if _EXTENDERS is None:
        symbols = _python_symbols(_python_files())
        raw = {}
        for rel in _python_files():
            targets = set()
            for name in _extends(rel):
                for other in symbols.get(name, ()):
                    if other != rel:
                        targets.add(other)
            if targets:
                raw[rel] = targets
        # A BASE EVERYTHING INHERITS IS NOT EVIDENCE ABOUT ONE LANE, the same
        # argument that makes a whole-surface registry a sink. Measured on this
        # tree 2026-09-16: `python/mojolearn/_mode.py` is extended by 23
        # package files and the next most-extended is extended by 2, so every
        # lane reached _mode.py, pulled in all 23, and the median file went
        # from 32 lanes to 197. Nothing is lost by dropping it: a change to
        # _mode.py itself still selects everything that imports it, and an
        # extender is attributed by its own ordinary edges.
        fan = {}
        for targets in raw.values():
            for rel in targets:
                fan[rel] = fan.get(rel, 0) + 1
        universal = {rel for rel, n in fan.items() if n > UNIVERSAL_BASE_MAX_EXTENDERS}
        out = {}
        for rel, targets in raw.items():
            kept = targets - universal
            if kept:
                out[rel] = kept
        _EXTENDERS = out
    return _EXTENDERS


def _extending_files(closure):
    """Every package file that extends something in `closure`, transitively."""
    edges = extenders()
    out, changed = set(), True
    while changed:
        changed = False
        for rel, targets in edges.items():
            if rel in out or rel in closure:
                continue
            if targets & (closure | out):
                out.add(rel)
                changed = True
    return out


def _python_closure(seeds, sinks=()):
    """The package's own import closure of `seeds`. A sink is included but not
    followed: the walk stops at a registry rather than stepping through it into
    every other family."""
    out, stack = set(), list(seeds)
    while True:
        while stack:
            rel = stack.pop()
            if rel in out:
                continue
            out.add(rel)
            if rel not in sinks:
                stack.extend(_python_imports(rel))
        # THE EDGE THAT RUNS BACKWARDS. A file that subclasses or patches a
        # class in this closure can change what the lane computes even though
        # nothing in the closure imports it.
        #
        # ITS IMPORTS ARE FOLLOWED TOO. Leaving them out was tried first and
        # the whole-tree inversion check caught it: `mamba1` reached
        # `neural_inference.py` without reaching the `_samba_impl` and
        # `_transformer_impl` that file imports at module level, so "F imports
        # B, therefore every lane reaching F reaches B" stopped holding. An
        # invariant that holds over the WHOLE TREE is worth more than the few
        # files it costs, and it cost nothing measurable here: the median file
        # is 33 lanes either way.
        fresh = _extending_files(out) - out
        if not fresh:
            return out
        stack.extend(fresh)


def _mojo_module_files(dotted, rel):
    """Every repository file one Mojo import could mean, resolved against the
    repository root AND against the importing file's OWN DIRECTORY.

    THE IMPORTER'S DIRECTORY IS ON THE INCLUDE PATH. Every binding is built
    with `-I . -I bindings` (bindings/build_rf.sh:123, build_trees.sh:134,
    build_gbdt.sh:268), so `bindings/_mojolearn_rf.mojo` writing
    `from forest_inference_binding import forest_prepare_gpu_binding` means
    `bindings/forest_inference_binding.mojo`, which is compiled into the
    shipped `.so`. Resolving against the root alone found no such file and
    dropped the import as one of the toolchain's own, so the whole resident
    forest inference tree (`bindings/forest_inference_binding.mojo` ->
    `core/forest_inference_model.mojo` -> `core/forest_inference.mojo`) was in
    no lane's map while being a shipped surface.

    This is not a new rule for this tree: `tools/bincache.py:198`, written for
    the binding cache and against the same compiler, searches
    `list(roots) + [importer.parent]` for exactly this reason.

    `from max.gpu.host import ...` still resolves to nothing under either root
    and is still dropped, which is what makes the toolchain fall out."""
    parts = dotted.split(".")
    out = set()
    for base in ("", os.path.dirname(rel)):
        for cand in (os.path.join(base, *parts) + ".mojo",
                     os.path.join(base, *parts, "__init__.mojo")):
            if os.path.exists(os.path.join(ROOT, cand)):
                out.add(cand)
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
        out |= _mojo_module_files(m.group(1) or m.group(2), rel)
    return out


_MOJO_TRAIT_RE = re.compile(r"^\s*trait\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
_MOJO_STRUCT_RE = re.compile(
    r"^\s*struct\s+[A-Za-z_0-9]+(?:\[[^\]]*\])?\s*\(([^)]*)\)", re.M)
_MOJO_MAIN_RE = re.compile(r"^\s*(?:fn|def)\s+main\s*\(", re.M)


@_by_path
def is_standalone_program(rel):
    """Does this Mojo source define its own `main`. Such a file is compiled on
    its own and linked into no binding, so it cannot change what a lane
    computes however much of the tree it names."""
    return bool(_MOJO_MAIN_RE.search(_read(rel)))


_MOJO_CONFORMANCE = None


def mojo_conformance_edges():
    """(conforming file, trait, declaring file) for every REPO trait.

    THE MOJO ANALOGUE OF THE PYTHON SUBCLASS EDGE, and the reason it does not
    need to be followed. In Python a subclass REPLACES behaviour for anyone who
    constructs it, so the edge runs backwards along imports. In Mojo a struct
    conforming to a trait is reached only when something parametrises on that
    trait AND IS HANDED THAT STRUCT BY NAME, and naming a symbol from another
    file requires importing it. So the dispatcher already imports the
    implementation and the forward walk already has it.

    Two exemptions, both DERIVED and both checked on this tree rather than
    asserted:

      * a file with its own `main` is a standalone program. All five files
        that conform to a repo trait purely to TEST it have one, and none of
        the seven shipped implementations does.
      * the conforming file must actually IMPORT the trait's declaration.
        `core/philox.mojo` and `mamba/host/gen/philox.mojo` each declare their
        OWN `U32Stream` and neither imports the other, so matching the trait
        by name alone invented an edge between two unrelated generators and
        claimed 80 missing lanes."""
    global _MOJO_CONFORMANCE, _MOJO_IMPORTERS
    if _MOJO_CONFORMANCE is None:
        mojo = sorted(f for f in tracked_files() if f.endswith(".mojo"))
        declared = {}
        for rel in mojo:
            for m in _MOJO_TRAIT_RE.finditer(_read(rel)):
                declared.setdefault(m.group(1), set()).add(rel)
        out = []
        for rel in mojo:
            if is_standalone_program(rel):
                continue
            imports = _mojo_imports(rel)
            for m in _MOJO_STRUCT_RE.finditer(_read(rel)):
                for base in m.group(1).split(","):
                    name = base.strip().split("[")[0].strip()
                    for decl in declared.get(name, ()):
                        if decl != rel and decl in imports:
                            out.append((rel, name, decl))
        _MOJO_CONFORMANCE = sorted(set(out))
    return _MOJO_CONFORMANCE


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
        files = _mojo_module_files(m.group(1), rel)
        if not files:
            continue                         # the toolchain's own modules, not ours
        for name in m.group(2).split(","):
            # THE LOCAL NAME, WHICH IS THE ALIAS WHEN THERE IS ONE. An export's
            # body is searched for these names, and the body writes what the
            # import BOUND. `bindings/_mojolearn_metrics.mojo:56` is
            # `from umap.estimator import fit_transform as umap_fit_transform`
            # and `umap_fit_transform_binding` calls `umap_fit_transform`.
            # Recording `fit_transform` made `\bfit_transform\b` miss it (the
            # underscore before `fit` is a word character), so the whole umap
            # tree was invisible to the per-export scan and reached the `umap`
            # lane only because the metrics HOST family happens to list
            # umap/graph.mojo among its host modules. `par-graph-umap`, which
            # runs the same fit across devices and is not in that family, was
            # credited with none of it.
            parts = [p.strip() for p in name.strip().split(" as ")]
            local = parts[-1] if len(parts) > 1 else parts[0]
            if local and local != "*":
                out.setdefault(local, set()).update(files)
    return out


@_by_path
def _binding_exports(rel):
    """export name -> the binding function implementing it, from the
    `module.def_function[impl]("name")` registrations a binding ends with.

    THE IMPL MAY BE PARAMETRIZED. `def_function[forest_prepare_gpu_binding[True]]`
    and `def_function[rf_classifier_fit_binding[False]]` are the ordinary
    spelling in the forest bindings, and requiring a bare identifier dropped 35
    exports across five bindings, among them `rf_classifier_fit`,
    `et_classifier_fit` and `forest_prepare_gpu`. A dropped export is not a
    wide answer: the lane still HITS other exports, so the per-export branch
    runs and the dropped export's whole tree is simply absent."""
    try:
        text = _read(rel)
    except OSError:
        return {}
    return {m.group(2): m.group(1) for m in re.finditer(
        r"def_function\[\s*([A-Za-z0-9_]+)\s*(?:\[[^\[\]]*\])?\s*\]\s*\(\s*\"([A-Za-z0-9_]+)\"", text)}


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
                    # THE EXPORT'S BODY IS NOT ONLY THE FUNCTION NAMED. An
                    # export reaches a Mojo file two ways this scan used to
                    # miss, and both are the ordinary spelling here:
                    #  * the impl is IMPORTED, not defined in the binding file.
                    #    `def_function[forest_prepare_gpu_binding[True]]` names
                    #    a function in bindings/forest_inference_binding.mojo,
                    #    so `blocks.get` returned "" and the export
                    #    contributed nothing at all.
                    #  * the impl calls a helper DEFINED IN THE SAME FILE which
                    #    is where the imported symbol appears.
                    #    `rf_predict_proba_gpu_parallel_binding` calls the
                    #    file-local `_rf_predict_gpu_parallel`, and only that
                    #    helper names `forest_predict_gpu`.
                    # Both are closed here: the export's name is resolved as an
                    # imported symbol in its own right, and the bodies are
                    # followed through the file's own functions.
                    bodies, seen, stack = [], set(), [exports[export]]
                    while stack:
                        name = stack.pop()
                        if name in seen:
                            continue
                        seen.add(name)
                        seeds |= syms.get(name, set())
                        body = blocks.get(name)
                        if body is None:
                            continue
                        bodies.append(body)
                        for other in blocks:
                            if other not in seen and re.search(
                                    r"\b%s\b" % re.escape(other), body):
                                stack.append(other)
                    body = "\n".join(bodies)
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


_REVERSE = None


def reverse_map(sources=None):
    """file -> the lanes that exercise it. The default map's inversion is
    memoized like the map itself (it was rebuilt on every corpus lookup, 30 s
    of the v0.8.8 selection); a caller handing in its own `sources` gets a
    fresh one."""
    global _REVERSE
    if sources is None:
        if _REVERSE is None:
            _REVERSE = reverse_map(lane_sources()[0])
        return _REVERSE
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


def _strip_docstrings(tree):
    """The same tree with every module, class and function docstring removed.

    Comments never reach an AST at all, and `ast.dump` without attributes
    carries no line numbers, so comparing two stripped dumps compares the CODE
    and nothing else. A body left empty becomes `pass`, which is what a
    function whose only statement was a docstring already did."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def code_dump(text):
    """One Python source as code alone, or None when it does not parse.

    ONLY DOCSTRINGS ARE DROPPED. A string literal that is not the first
    statement of a module, class or function is an ordinary value, appears in
    the dump, and a change to it is a change to the code. That is the line
    between "this edit cannot move a bit" and a diff heuristic on `#` and
    `\"\"\"`, which would call an edited error message inert."""
    try:
        return ast.dump(_strip_docstrings(ast.parse(text)))
    except SyntaxError:
        return None


def _git_show(ref, path):
    key = f"{ref}:{path}"
    if key not in _GIT_SHOW:
        try:
            _GIT_SHOW[key] = subprocess.run(["git", "-C", ROOT, "show", key],
                                            capture_output=True, text=True, check=True).stdout
        except subprocess.CalledProcessError:
            _GIT_SHOW[key] = None
    return _GIT_SHOW[key]


_GIT_SHOW = {}


@_by_path
def _hashes_a_file(rel):
    """True when this module builds a hashlib digest out of a file it opens in
    binary mode. Its own bytes, and the repository paths it names, are then
    hashed at run time."""
    if "hashlib" not in _read(rel):
        return False
    tree = _parse(rel)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name == "open" and any(isinstance(a, ast.Constant) and a.value == "rb"
                                      for a in node.args):
                return True
    return False


_SOURCE_HASHED = None


def source_hashed_files():
    """Files whose BYTES are read back and hashed while the library runs, so a
    change to a COMMENT in them moves a recorded value even though it moves no
    arithmetic. `python/mojolearn/_byte_lm_impl.py` publishes `source_sha256`
    over six named sources and over itself.

    These are the one exception to the docstring rule below, and they are
    DERIVED: the modules that hash a file are found by syntax, and the paths
    each one names by string literal are what it hashes. A hand-kept list here
    would rot exactly like the lane total did."""
    global _SOURCE_HASHED
    if _SOURCE_HASHED is None:
        out = set()
        for rel in _python_files():
            if not _hashes_a_file(rel):
                continue
            out.add(rel)
            tree = _parse(rel)
            for node in ast.walk(tree or ast.Module(body=[], type_ignores=[])):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and node.value.endswith((".py", ".mojo", ".sh"))
                        and os.path.isfile(os.path.join(ROOT, node.value))):
                    out.add(node.value)
        _SOURCE_HASHED = out
    return _SOURCE_HASHED


def docstring_only(ref, path):
    """Does this path differ from `ref` in docstrings and comments ALONE.

    False for anything this cannot prove: a file that is not Python, one that
    is new or deleted, one that does not parse on either side, and one whose
    bytes are hashed at run time. A Mojo docstring edit is not covered, because
    there is no parser for it here and guessing would be the whole point of the
    failure this avoids.

    THIS WIDENS WHAT RETURNS A NARROW ANSWER, which is the dangerous
    direction. Being wrong here turns a real code change into "nothing
    affected"; being wrong the other way only costs a sweep."""
    if not path.endswith(".py") or path in source_hashed_files():
        return False
    old = _git_show(ref, path)
    if old is None:
        return False
    try:
        new = _read(path)
    except OSError:
        return False
    a, b = code_dump(old), code_dump(new)
    return a is not None and b is not None and a == b


_TRACKED = None


def tracked_files():
    """Every tracked path, once."""
    global _TRACKED
    if _TRACKED is None:
        try:
            res = subprocess.run(["git", "-C", ROOT, "ls-files"],
                                 capture_output=True, text=True, check=True)
            _TRACKED = set(res.stdout.split("\n")) - {""}
        except subprocess.CalledProcessError:
            _TRACKED = set()
    return _TRACKED


TESTS = os.path.join(PKG, "tests") + os.sep

_IMPORT_OF = r"(?:^|\n)[^\S\n]*(?:from[^\S\n]+[.\w]*\b{0}\b|import[^\S\n]+[.\w]*\b{0}\b)"


def test_module_inert(path):
    """Why this test module cannot move a lane's bits, or None.

    `_python_files()` already leaves `python/mojolearn/tests/` out of the map,
    on the stated ground that a test cannot change what a lane computes. The
    selector never acted on that, so every file under it read NOT ATTRIBUTABLE
    and sent its lane to the full sweep: lane/kmeans-save ended at 212 partly
    because of `tests/test_host_model_kmeans.py`.

    The ground is only true while nothing outside the tests directory imports
    the module, so that is what is checked, over the whole tracked tree rather
    than over the lane corpus: an import from a build script or a tool would
    still be an import. A test importing another test proves nothing, because
    both are inert by the same argument.

    This is deliberately NOT routed through `unreachable`, which asks a
    different question and answers it conservatively: `python/mojolearn/` turns
    up inside an f-string in `_backend.py` (`' needs python/mojolearn/'`
    followed by a computed name), which is exactly the constructed-path shape
    that rule must refuse."""
    if not path.startswith(TESTS) or not path.endswith(".py"):
        return None
    module = os.path.basename(path)[:-3]
    if len(module) < 4:
        return None
    # NOT AN IMPORT LINE. Attacked 2026-09-16 with
    # `importlib.import_module('mojolearn.tests.test_host_model_kmeans')`,
    # which an import-statement regex cannot see and which was called inert.
    # The module NAME is searched instead, anywhere outside the tests
    # directory, so a dynamic import, a `python -m` line in a script and a
    # bare mention all count. Over-firing here only costs a sweep.
    #
    # ONLY FILES A LANE REACHES CAN VOTE (2026-09-21), the argument
    # `_reaching_corpus` makes for every other path. A pixi task that runs
    # `python -m mojolearn.tests.test_byte_lm_surface`, a workflow that lists
    # it, or `tools/transformer_transfer_check.py` importing a helper from a
    # test module does not put that module into any lane's process: none of
    # those files is in any lane's closure. Counting them sent 34 of 125 test
    # modules, and with them every release that touched a test, to the full
    # sweep. The search is still by NAME over the code (`_searchable`: every
    # string literal survives, docstrings and comments do not), so a dynamic
    # import from a file a lane does reach still counts.
    importers = []
    for rel in _reaching_corpus():
        if rel.startswith(TESTS) or _is_inert(rel) or rel in _NOT_CORPUS:
            continue
        if module in _searchable(rel):
            importers.append(rel)
    if importers:
        return None
    return ("a test module: it is outside the map by construction, because a test cannot change "
            "what a lane computes, and no file any lane reaches names it in code")


_MOJO_IMPORTERS = None


def _mojo_importers(path):
    """The tracked Mojo files that import `path`, by the same resolution the
    forward walk uses."""
    global _MOJO_IMPORTERS
    if _MOJO_IMPORTERS is None:
        out = {}
        for rel in tracked_files():
            if not rel.endswith(".mojo") or _is_inert(rel):
                continue
            if is_standalone_program(rel):
                # A FILE WITH ITS OWN `main` IS A PROGRAM, and a program is
                # compiled on its own and linked into no binding, so importing
                # something does not put it in any lane's way. The same
                # exemption the conformance edge needed, measured the same way.
                # Without it the three new umap/checks files, which import each
                # other, each made the others look reachable.
                continue
            for target in _mojo_imports(rel):
                out.setdefault(target, set()).add(rel)
        _MOJO_IMPORTERS = out
    return _MOJO_IMPORTERS.get(path, set())


def _reaching_corpus():
    """Every file some lane already reaches, plus the harness and the manifest.

    THIS IS THE WHOLE POINT OF THE NEXT FUNCTION. For a lane to reach a file,
    something IN THAT LANE'S CLOSURE has to name it, directly or through a
    chain. So the only files whose text can extend a lane's reach are the ones
    the map already contains. `pixi.toml` naming `umap/checks` does not put a
    check into any lane, and neither does a build task, a CI workflow or a
    contribution gate: none of them is in any lane's closure.

    PLUS EVERY BUILD SCRIPT UNDER bindings/ (2026-09-21). A lane runs the
    binary those scripts compile, so what a build script names (a helper it
    execs, a generator it runs, a define file it reads) reaches the lane at
    build time even though no lane's Python or Mojo closure contains the
    script. The host shims were already in the map; the GPU build scripts and
    `build_host_family.sh` were not, which was harmless only while every file
    under tools/ read as reachable through a directory join. Adding them here
    keeps `tools/with_build_lock.sh` and `tokenizer/tools/gen_unicode_table.sh`
    reachable now that a literal `join(base, "tools", "x.py")` no longer
    counts as walking tools/."""
    global _CORPUS
    if _CORPUS is None:
        builds = {rel for rel in tracked_files()
                  if rel.startswith("bindings" + os.sep) and rel.endswith(".sh")}
        _CORPUS = sorted(set(reverse_map()) | {HARNESS, MANIFEST} | builds)
    return _CORPUS


_CORPUS = None
_CORPUS_TEXT = None


def _corpus_text():
    """Every searchable corpus file's code, joined, so a token that occurs
    NOWHERE is dismissed with one substring test instead of one regex per
    file. Measured 2026-09-21 on the 2,329-path v0.8.8 diff: the per-file
    search was 74 s of a 139 s selection. A token that does occur is still
    searched file by file with its exact pattern, so the answer is unchanged."""
    global _CORPUS_TEXT
    if _CORPUS_TEXT is None:
        _CORPUS_TEXT = "\n\0\n".join(_searchable(rel) for rel in _reaching_corpus()
                                     if not _is_inert(rel) and rel not in _NOT_CORPUS)
    return _CORPUS_TEXT


@_by_path
def _searchable(rel):
    """One corpus file with its DOCSTRINGS AND COMMENTS removed, because a
    path written in prose is not a reference to it.

    Measured 2026-09-16: `umap/` appears in `bindings/_mojolearn_metrics.mojo`
    ("recorded in umap/README.md"), in a 1,600-line ROUTING docstring in
    `checks/kernel_matrix.mojo`, and in a comment in `_backend.py`. All three
    are sentences. Counting them made three new files under `umap/checks/`
    look reachable and sent a diff that changed no product code to all 212
    lanes.

    Python goes through the same AST dump the docstring rule uses, so every
    string that is NOT a docstring survives and a path held in a constant
    still counts. Mojo has no parser here, so a triple-quoted block is dropped
    only when it OPENS a line, which is the docstring shape; a triple-quoted
    string on an assignment is left alone, because stripping too much here
    would call a reachable file unreachable, and that is the direction that
    costs a defect."""
    try:
        text = _read(rel)
    except OSError:
        return ""
    if rel.endswith(".py"):
        return code_dump(text) or text
    if rel.endswith(".mojo"):
        out, in_doc = [], False
        for line in text.splitlines():
            stripped = line.lstrip()
            if in_doc:
                if '"""' in line:
                    in_doc = False
                continue
            if stripped.startswith('"""'):
                if not (stripped.rstrip().endswith('"""') and len(stripped.rstrip()) > 5):
                    in_doc = True
                continue
            out.append(line.split("#", 1)[0])
        return "\n".join(out)
    if rel.endswith(".sh"):
        # A FULL-LINE SHELL COMMENT is prose. Only whole lines are dropped:
        # `#` inside a line is also `${x#y}` and `"#"`, which are code.
        return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return text


def _named_by_the_corpus(token, skip=(), whole=False):
    """The files a lane reaches that name `token` in CODE, as PATHS, never a
    count: a grep that prints a number cannot be told from one that failed.

    `whole` is for a DIRECTORY token, where a bare substring search is the
    wrong question. `umap/` occurs in every reference to every file under
    `umap/`, so searching for it asks "does anything use this tree", which is
    always yes. What matters is whether anything names the DIRECTORY ITSELF,
    which is the shape a glob or a directory walk takes: `umap/` followed by a
    quote, a star or a bracket rather than by another path component."""
    # A DIRECTORY TOKEN IS ANCHORED ON THE LEFT TOO (2026-09-21): `checks/`
    # inside "(mamba/checks/)" names mamba/checks/, a different directory, and
    # was what kept every new file under the top-level checks/ reachable. A
    # token preceded by `<word>/` is a subdirectory of something else and does
    # not count; one preceded by `/` alone (ROOT + "/checks/") or `../` does.
    if token not in _corpus_text():
        return []
    pattern = re.compile((r"(?<![A-Za-z0-9_-]/)" if whole else "") + re.escape(token)
                         + (r"(?![A-Za-z0-9_.])" if whole else ""))
    out = []
    for rel in _reaching_corpus():
        if rel in skip or _is_inert(rel) or rel in _NOT_CORPUS:
            continue
        if pattern.search(_searchable(rel)):
            out.append(rel)
    return out


_JOINERS = frozenset({"join", "Path", "PurePosixPath", "open", "glob", "iglob", "rglob"})


@_by_path
def _path_join_roots(rel):
    """String constants this Python file hands to a path-building call as its
    FIRST argument: `os.path.join('bench', name)` yields 'bench'. A directory
    named this way is being walked or built on, even though its name never
    appears with a slash.

    A CALL THAT SPELLS ONE FILE WALKS NOTHING (2026-09-21). In
    `os.path.join(base, "tools", "identity_break.py")` every argument from the
    first constant on is a constant and the last one is a file name, so the
    call names exactly `tools/identity_break.py`. That file is still found by
    its basename, which the token search reads; what is dropped is the claim
    that the call walks `tools/`. Counting it as a walk made every one of the
    ~250 files under tools/ reachable from `_identity.py` and `_verify.py`,
    and every leg script, benchmark and test there sent a release to all
    lanes. A join whose tail is a variable (`join('armprobedir', name)`) still
    counts: that one really does build paths under the directory."""
    tree = _parse(rel)
    out = set()
    for node in ast.walk(tree or ast.Module(body=[], type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name not in _JOINERS:
            continue
        consts = [k for k, a in enumerate(node.args)
                  if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if consts and len(node.args) > 1 and all(
                isinstance(a, ast.Constant) and isinstance(a.value, str)
                for a in node.args[consts[0]:]) \
                and re.search(r"[A-Za-z0-9_]\.[A-Za-z0-9]{1,5}$", node.args[-1].value):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.add(arg.value.strip("/"))
            elif isinstance(arg, ast.BinOp) and isinstance(arg.left, ast.Constant) \
                    and isinstance(arg.left.value, str):
                out.add(arg.left.value.strip("/"))
    return out


def _joined_with(directory, skip=()):
    """The corpus files that hand `directory` to a path-building call."""
    return [rel for rel in _reaching_corpus()
            if rel.endswith(".py") and rel not in skip and not _is_inert(rel)
            and rel not in _NOT_CORPUS and directory in _path_join_roots(rel)]


def unreachable(path):
    """Why NOTHING can reach `path`, or None when something might.

    Two halves, and both are needed. The map alone is not proof, because the
    map is a model and an unattributable path falls back precisely because a
    model can be incomplete. The second half closes that: it searches the files
    the map DOES contain for this path, for its stem, and for EVERY ANCESTOR
    DIRECTORY in both path and dotted form. A fixture read by a glob is not
    named by its own name, but the directory the glob walks is, and that
    directory would have to be named by something a lane reaches.

    Refused for anything in the Python package or under `bindings/`, which are
    resolved by name at load time, and for the harness, the manifest, the
    registries and the selection machinery, each of which has its own rule.

    There is no need to exclude the rest of the same change: the corpus is the
    files a lane ALREADY reaches, so two new files cannot make each other
    reachable, and a MODIFIED file that a lane does reach must keep its vote.
    An earlier spelling passed the whole changed list as a skip set, which
    would have hidden exactly the case that matters: a new source added
    together with the import that pulls it in."""
    if path in (HARNESS, MANIFEST) or path in _NOT_CORPUS or path in enumerator_files():
        return None
    if path in reverse_map():
        return None
    if path.startswith(PKG + os.sep):
        return _package_file_unreachable(path)
    if path.startswith("bindings" + os.sep):
        return None
    if path.endswith(".mojo") and not _mojo_importers(path):
        why = _unimported_mojo_unreachable(path)
        if why:
            return why
    if path.endswith(".py"):
        return _outside_python_unreachable(path)
    if path.endswith(".mojo") and _mojo_importers(path):
        # IMPORTED BY SOMETHING, EVEN SOMETHING THE MAP DOES NOT HAVE. The
        # corpus is what a lane reaches, and a chain of files the map is
        # missing votes nowhere: `core/forest_inference_model.mojo` is
        # imported by `bindings/forest_inference_binding.mojo`, which is
        # itself outside the map, so nothing in the corpus named either and
        # the model file read "nothing reaches it". A Mojo import is a
        # compile-time fact and does not need the corpus to be believed.
        return None
    stem = os.path.basename(path)
    tokens = [(path, False), (stem, False)]
    if "." in stem:
        tokens.append((stem.rsplit(".", 1)[0], False))
    parts = path.split(os.sep)[:-1]
    for k in range(len(parts)):
        d = os.sep.join(parts[:k + 1])
        tokens.append((d + os.sep, True))
        if k:
            # The DOTTED form only from the second level down. A one-word
            # top-level directory is not a path, it is a word: `umap` matches
            # `from . import umap`, the package module of the same name, and a
            # `Constant(value='umap')` in a family table. `umap.checks` is a
            # module path and means what it says.
            tokens.append((d.replace(os.sep, "."), True))
    tokens = [(t, w) for t, w in dict.fromkeys(tokens) if len(t) >= 4]
    if (path, False) not in tokens:
        return None                      # too short a path to search for honestly
    for token, whole in tokens:
        hits = _named_by_the_corpus(token, skip={path}, whole=whole)
        if hits:
            return None
    for part in dict.fromkeys(parts):
        # A DIRECTORY NAMED WITHOUT ITS SLASH, as a bare string handed to a
        # path join. Attacked 2026-09-16 with
        # `os.path.join('armprobedir', name + '.bin')`: the slash tokens above
        # see nothing and the file was called unreachable. Searched
        # STRUCTURALLY rather than as text, because the bare word `umap` is
        # also the name of a package module and of a family-table entry, and
        # matching those would send every new file under umap/ to a sweep.
        if _joined_with(part, skip={path}):
            return None
    return ("nothing reaches it: no lane's derived source set contains this path, and no file "
            "that any lane DOES reach names it, its stem or any directory above it")


def _named_in_code(patterns, skip=(), literals=()):
    """Corpus files whose searchable text matches any of `patterns`, a dict
    of corpus-file suffix -> compiled regex list ("" is the default). When
    every pattern can only match text containing one of `literals`, a
    literal absent from the whole corpus answers at once."""
    if literals and not any(lit in _corpus_text() for lit in literals):
        return []
    out = []
    for rel in _reaching_corpus():
        if rel in skip or _is_inert(rel) or rel in _NOT_CORPUS:
            continue
        kind = os.path.splitext(rel)[1]
        text = _searchable(rel)
        if any(p.search(text) for p in patterns.get(kind, patterns[""])):
            out.append(rel)
    return out


def _module_patterns(stem, dotted=()):
    """How a Python module is named when something LOADS it, per corpus kind.

    In a Python corpus file (read as its code dump, docstrings gone) an import
    of `stem` is `alias(name='stem')` or `ImportFrom(module='x.stem')` and a
    dynamic import is `Constant(value='stem')`: a quoted string that IS the
    name or ends in `.stem`. A Mojo file can only reach a Python module through
    `Python.import_module("stem")`, a quoted name again. A shell script runs it
    as `-m x.stem`, by path, or imports it in a heredoc, so there `-m`,
    `import` and `from` followed by the dotted name count. What does NOT
    count is the stem inside a longer word or a sentence: `stage` in "no stage
    the card records", `setup` in "trainer_setup", `wheel` in an error message.
    Those were the hits that kept packaging/portable_math/stage.py and
    python/setup.py in every release's full sweep."""
    word = re.escape(stem)
    quoted = re.compile(r"""['"](?:[\w.]*\.)?%s['"]""" % word)
    run = re.compile(r"(?:-m|import|from)\s+(?:\w+\.)*%s(?![\w])" % word)
    out = {"": [run], ".py": [quoted], ".mojo": [quoted]}
    for d in dotted:
        out[""].append(re.compile(re.escape(d)))
        out[".py"].append(re.compile(re.escape(d)))
    return out


def _outside_python_unreachable(path):
    """A Python file outside the package (a tool, a check, a packaging script)
    that no lane reaches.

    The same corpus search as `unreachable`, with ONE difference: the bare
    module stem counts only where something could LOAD the module
    (`_module_patterns`), not wherever the word occurs. The path, the basename
    `stem.py`, every ancestor directory and every structural directory join
    are searched exactly as before."""
    stem = os.path.basename(path)[:-3]
    dotted = path[:-3].replace(os.sep, ".")
    tokens = [(path, False), (os.path.basename(path), False)]
    parts = path.split(os.sep)[:-1]
    for k in range(len(parts)):
        d = os.sep.join(parts[:k + 1])
        tokens.append((d + os.sep, True))
        if k:
            tokens.append((d.replace(os.sep, "."), True))
    for token, whole in dict.fromkeys(tokens):
        if len(token) >= 4 and _named_by_the_corpus(token, skip={path}, whole=whole):
            return None
    if len(stem) < 3 or _named_in_code(_module_patterns(stem, (dotted,)), skip={path}, literals=(stem,)):
        return None
    for part in dict.fromkeys(parts):
        if _joined_with(part, skip={path}):
            return None
    return ("nothing reaches it: a Python file outside the package that no lane's source set "
            "contains, that nothing a lane reaches imports or names by path, and whose "
            "directory nothing a lane reaches walks")


def _mojo_packages_built():
    """Do any build scripts compile a whole DIRECTORY (`mojo package`)? Then a
    Mojo file nobody imports can still be in a binary, and the rule below must
    not fire."""
    return any("mojo package" in _read(rel) for rel in _reaching_corpus()
               if rel.startswith("bindings" + os.sep) and rel.endswith(".sh"))


def _unimported_mojo_unreachable(path):
    """A Mojo source that no other tracked Mojo source imports.

    A Mojo file reaches a lane only by being compiled into a binding: named on
    a `mojo build` line (every such line is in a bindings/ build script, which
    is in the corpus) or imported by something that is. Nothing imports this
    one (`_mojo_importers` is a compile-time fact over the whole tree, the same
    resolution the forward walk uses), so the only way in is by NAME: its path,
    its basename, its stem or its dotted module path, in a file a lane reaches.

    The DIRECTORY tokens `unreachable` also tries are not used here. They find
    sentences in string literals ("defines are for the lane gates
    (transformer/checks/)") and an unrelated `join(tmp, "llama")`, and a
    directory cannot pull an unimported Mojo file into a binary unless a build
    compiles the whole directory, which `_mojo_packages_built` rules out.
    This is what keeps a new benchmark or check program under */checks/ or
    tools/ from sending a release to every lane."""
    if _mojo_packages_built():
        return None
    stem = os.path.basename(path)[:-5]
    dotted = path[:-5].replace(os.sep, ".")
    for token in dict.fromkeys((path, os.path.basename(path), dotted)):
        if _named_by_the_corpus(token, skip={path}):
            return None
    if len(stem) < 4 or _named_in_code({"": [re.compile(r"(?<![\w])%s(?![\w])" % re.escape(stem))]},
                                       skip={path}, literals=(stem,)):
        return None
    return ("nothing reaches it: a Mojo source that no tracked Mojo source imports, so it is in no "
            "binding, and that nothing a lane reaches names by path, basename, stem or module path")


_SHELL_CALLS = frozenset({"system", "popen", "getoutput", "getstatusoutput"})


@_by_path
def _runs_shell_strings(rel):
    tree = _parse(rel)
    for node in ast.walk(tree or ast.Module(body=[], type_ignores=[])):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in _SHELL_CALLS:
                return True
            for kw in node.keywords:
                if kw.arg == "shell" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                    return True
    return False


def _corpus_runs_shell_strings():
    """Does any Python file a lane reaches hand a command STRING to a shell
    (`shell=True`, `os.system`, `os.popen`, `subprocess.getoutput`)? If none
    does, a command written inside a Python string cannot be executed, and
    the string is a message."""
    return any(_runs_shell_strings(rel) for rel in _reaching_corpus() if rel.endswith(".py"))


_PKG_IMPORT_CLOSURE = None


def _package_import_closure():
    """Every package module the reaching corpus can IMPORT, walking straight
    through the registries (`__init__.py` included) that the lane map stops
    at. A module `import mojolearn` executes is in here even when no lane's
    door calls into it, because its top-level code runs in every lane's
    process."""
    global _PKG_IMPORT_CLOSURE
    if _PKG_IMPORT_CLOSURE is None:
        seeds = [rel for rel in _reaching_corpus() if rel.startswith(PKG + os.sep) and rel.endswith(".py")]
        seeds.append(os.path.join(PKG, "__init__.py"))
        _PKG_IMPORT_CLOSURE = _python_closure(seeds, sinks=())
    return _PKG_IMPORT_CLOSURE


def _package_file_unreachable(path):
    """A file inside python/mojolearn/ that no lane can load.

    The package used to be refused outright because its modules are resolved
    by name at load time. That stays true for everything the rule below does
    not prove, and it proves little:

      * a TOP-LEVEL MODULE (python/mojolearn/X.py) is unreachable when no
        lane's closure has it, the corpus cannot import it even through the
        registries (`_package_import_closure`), and no corpus file names it
        where a loader would (`_module_patterns`). `__main__.py` is searched
        as `mojolearn.__main__`, `-m mojolearn` and `run_module`, because the bare
        `__main__` is in every script guard. These are the verifier's own
        modules (`_verify_par.py`, `_verify_reference.py`, `cross_vendor.py`):
        they run from `python -m mojolearn verify`, never inside a lane.
      * a PACKAGE DATA FILE that is not code (.json, .pdf, .csv) is
        unreachable when no corpus file names its basename or its path under
        the package and no corpus file walks its own directory. The package
        root itself (`python`, `mojolearn`) is joined by every file that puts
        the package on sys.path, so those two parts are not read as walks.
        `verify_reference/table.json` is the case: the reference table the
        verifier compares a finished column with, which no lane reads.

    Subpackages (`models/`, `tests/`) are left to the rules that already
    handle them, because `_python_imports` does not resolve a dotted relative
    import into a subpackage and this rule would then be guessing."""
    rel = path[len(PKG) + 1:]
    if os.sep in rel:
        if rel.startswith("tests" + os.sep) or not path.endswith((".json", ".pdf", ".csv")):
            return None
    if path.endswith(".py"):
        if path in _package_import_closure() or path == os.path.join(PKG, "__init__.py"):
            return None
        stem = os.path.basename(path)[:-3]
        if stem == "__main__":
            # `python -m mojolearn` in a Python file is a usage string in a
            # message unless something runs command STRINGS. So in Python the
            # list form (`[sys.executable, "-m", "mojolearn", ...]`) counts,
            # and the string form counts only if some corpus file runs shell
            # strings at all (`_corpus_runs_shell_strings`).
            pats = [re.compile(re.escape(t)) for t in ("mojolearn.__main__", "run_module", "__main__.py")]
            listed = re.compile(r"Constant\(value='-m'\), Constant\(value='mojolearn'\)")
            string = re.compile(r"-m\s+mojolearn(?![\w.])")
            py = pats + [listed] + ([string] if _corpus_runs_shell_strings() else [])
            patterns = {"": pats + [string], ".py": py, ".mojo": pats + [string]}
        else:
            patterns = _module_patterns(stem, ("mojolearn." + stem,))
        if _named_in_code(patterns, skip={path}, literals=(stem,) if stem != "__main__" else ()):
            return None
        return ("nothing reaches it: a package module that no lane's closure contains, that "
                "nothing a lane reaches can import (even through __init__ and the registries), "
                "and that nothing a lane reaches names where a loader would")
    if not path.endswith((".json", ".pdf", ".csv")):
        return None
    for token in dict.fromkeys((path, rel, os.path.basename(path))):
        if len(token) >= 5 and _named_by_the_corpus(token, skip={path}):
            return None
    for part in rel.split(os.sep)[:-1]:
        if _named_by_the_corpus(part + os.sep, skip={path}, whole=True) or _joined_with(part, skip={path}):
            return None
    return ("nothing reaches it: package data that no file a lane reaches names by basename or "
            "path, in a directory no file a lane reaches walks")


_CONSTANTS = None


def module_constants():
    """Module-level `NAME = "literal"` bindings across the package, so a key
    written as `_KMEANS_FORMAT` can be compared with one written as
    `"mojolearn-kmeans-1"`. Best effort: a name that does not resolve is left
    unresolved and simply has to be new."""
    global _CONSTANTS
    if _CONSTANTS is None:
        out = {}
        for rel in _python_files():
            tree = _parse(rel)
            for node in (tree.body if tree else []):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            out.setdefault(target.id, node.value.value)
        _CONSTANTS = out
    return _CONSTANTS


def _key_value(node):
    """What a dict key resolves to, or None when it cannot be resolved."""
    if isinstance(node, ast.Constant):
        return ("const", node.value)
    if isinstance(node, ast.Name):
        value = module_constants().get(node.id)
        return ("const", value) if value is not None else ("name", node.id)
    return None


def _stmt_key(node):
    """A statement's IDENTITY, so two revisions can be aligned without using
    line numbers. An assignment is keyed by the name it binds, so a registry
    table that GREW is the same statement rather than a different one."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 \
            and isinstance(node.targets[0], ast.Name):
        return ("assign", node.targets[0].id)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ("def", node.name)
    return ("stmt", ast.dump(node))


def _container_delta(old, new):
    """The elements ADDED to, and CHANGED in, a container literal, or None when
    the change is not of that shape.

    Every old element must still be there, IN ORDER. A removal, a reorder or a
    replaced key is not an addition and gets nothing from this."""
    if type(old) is not type(new):
        return None
    if isinstance(old, ast.Dict):
        okeys = [ast.dump(k) for k in old.keys]
        nkeys = [ast.dump(k) for k in new.keys]
        kept = [k for k in nkeys if k in okeys]
        if kept != okeys or len(set(okeys)) != len(okeys):
            return None
        # TWO KEYS CAN BE DIFFERENT EXPRESSIONS AND THE SAME VALUE, and then
        # the later one OVERRIDES the earlier and an existing lane's dispatch
        # moves. Attacked 2026-09-16 with `"mojolearn-dbscan-" + "1"` beside
        # `"mojolearn-dbscan-1"`, which was admitted as an addition. A key must
        # be a literal or a bare name, and the values they resolve to must be
        # distinct.
        if any(not isinstance(k, (ast.Constant, ast.Name)) for k in new.keys):
            return None
        resolved = [_key_value(k) for k in new.keys]
        seen = [v for v in resolved if v is not None]
        if len(set(seen)) != len(seen):
            return None
        oval = dict(zip(okeys, old.values))
        added, changed = [], []
        for key, value in zip(nkeys, new.values):
            if key not in oval:
                added.append((key, value))
            elif ast.dump(oval[key]) != ast.dump(value):
                changed.append((key, value))
        return added, changed
    if isinstance(old, (ast.Tuple, ast.List, ast.Set)):
        odumps = [ast.dump(e) for e in old.elts]
        ndumps = [ast.dump(e) for e in new.elts]
        kept = [d for d in ndumps if d in odumps]
        if kept != odumps:
            return None
        return [(None, e) for e, d in zip(new.elts, ndumps) if d not in odumps], []
    return None


def _admissible_addition(node, old_keys, corpus):
    """May this brand-new top-level statement be admitted.

    An added statement at module level can mutate a registry, shadow a name or
    run a decorator, so only three shapes are allowed: an import of a module
    some lane ALREADY reaches, so nothing new is pulled into the process; an
    undecorated def with a new name; and an undecorated class with a new name
    whose body is only a docstring, defs and assignments of names. A class body
    executes when it is defined, which is why its contents are checked rather
    than assumed."""
    if isinstance(node, ast.ImportFrom):
        if not node.module:
            return False
        cand = os.path.join(PKG, node.module.split(".")[0] + ".py")
        return cand in corpus
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return not node.decorator_list and ("def", node.name) not in old_keys
    if isinstance(node, ast.ClassDef):
        if node.decorator_list or ("def", node.name) in old_keys:
            return False
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Pass)):
                continue
            if isinstance(item, ast.Assign) and all(isinstance(t, ast.Name) for t in item.targets):
                continue
            return False
        return True
    return False


def _lanes_named_by(nodes, sources, rev):
    """The lanes the added or changed material can reach: the lanes that reach
    the file defining each identifier it names, plus any lane it names outright,
    plus any lane whose name occurs inside a string it carries.

    Returns None when a name cannot be placed at all, because an addition this
    cannot attribute is an addition this must not narrow."""
    # A NAME DEFINED IN A REGISTRY SAYS NOTHING ABOUT ONE LANE. `_HostBound` is
    # defined in `_classical_host.py` itself, and that file is reached by every
    # lane by construction, so resolving the new class's base through it
    # attributed the addition to all 212. Registries are dropped from this
    # resolution for exactly the reason their edges are dropped from the map.
    sinks = enumerator_files() | {MANIFEST}
    symbols = {name: files - sinks for name, files in
               _python_symbols(_python_files()).items()}
    every = set(sources)
    lanes, unknown = set(), []
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) or isinstance(sub, ast.Attribute):
                name = getattr(sub, "id", None) or getattr(sub, "attr", None)
                if name in every:
                    lanes.add(name)
                    continue
                files = symbols.get(name)
                if files:
                    for rel in files:
                        lanes |= rev.get(rel, set())
                elif name and not name.startswith("__"):
                    unknown.append(name)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                text = sub.value
                if text in every:
                    lanes.add(text)
                    continue
                hit = {n for n in every if n in text}
                if hit:
                    lanes |= hit
                elif os.path.isfile(os.path.join(ROOT, text)):
                    lanes |= rev.get(text, set())
    return sorted(lanes) if lanes else None


def registry_lanes(ref, path, sources=None):
    """The lanes an ADDITIVE change to a whole-surface registry can reach, or
    None for every lane.

    `host_surface.py` and `_classical_host.py` name the entire binding surface,
    so the map drops their per-lane edges and any change to them selects
    everything. That is right for an arbitrary edit and wrong for an addition
    that names one estimator: lane/kmeans-save added an import, a `HostKMeans`
    class and one `_FORMATS` entry, and got 212 of 212.

    ADDITIVE IS VERIFIED. Every old top-level statement must still be present
    and in order, either byte for byte or as the same assignment whose
    container literal only GREW or whose value changed under an unchanged key.
    Everything else, including a removal, a reorder, an edited function body or
    a new bare statement, returns None."""
    if sources is None:
        sources, _ = lane_sources()
    rev = reverse_map(sources)
    old_text = _git_show(ref, path)
    if old_text is None:
        return None
    try:
        old_tree = _strip_docstrings(ast.parse(old_text))
        new_tree = _strip_docstrings(ast.parse(_read(path)))
    except (OSError, SyntaxError):
        return None
    old_keys = [_stmt_key(n) for n in old_tree.body]
    new_keys = [_stmt_key(n) for n in new_tree.body]
    if len(set(old_keys)) != len(old_keys) or len(set(new_keys)) != len(new_keys):
        return None                         # a repeated key cannot be aligned honestly
    kept = [k for k in new_keys if k in set(old_keys)]
    if kept != old_keys:
        return None                         # something was removed or reordered
    old_by_key = dict(zip(old_keys, old_tree.body))
    touched = []
    for key, node in zip(new_keys, new_tree.body):
        if key not in old_by_key:
            if not _admissible_addition(node, set(old_keys), set(rev)):
                return None
            touched.append(node)
            continue
        before = old_by_key[key]
        if ast.dump(before) == ast.dump(node):
            continue
        if key[0] != "assign":
            return None                     # an edited body reaches anything
        delta = _container_delta(before.value, node.value)
        if delta is None:
            return None
        added, changed = delta
        for element_key, value in added + changed:
            touched.append(value)
            if element_key is not None:
                # THE KEY IS ATTRIBUTION TOO. A registry keyed by lane name
                # says which lane an entry is about more directly than its
                # value does.
                touched.append(_key_node(before, node, element_key))
    return _lanes_named_by([t for t in touched if t is not None], sources, rev)


def _key_node(before, after, dumped_key):
    """The key expression matching `dumped_key`, so the key of a changed entry
    is attributed as well as its value. A table keyed by lane name is the
    common case and the key is the whole of the attribution."""
    for node in (after.value, before.value):
        for k in getattr(node, "keys", []) or []:
            if ast.dump(k) == dumped_key:
                return k
    return None


def _harness_segments(text):
    """`identity_break.py` as (lane -> its dumped definition) and the dumped
    list of every OTHER top-level statement.

    Positions are out by construction, and that is the point. The first
    spelling of this keyed a bare statement by its LINE NUMBER, so inserting
    one new lane renumbered every statement below it, every key changed, and a
    diff that touched no existing lane body answered "every lane". Measured
    2026-09-16 by lane/data-ordering-determinism: one additive hunk, true blast
    radius three lanes, selector said 212.

    Every `@lane(...)` name on a definition is recorded, not just the last, so
    a stacked registration cannot hide one."""
    tree = _strip_docstrings(ast.parse(text))
    lanes, others = {}, []
    for node in tree.body:
        decorators = getattr(node, "decorator_list", [])
        names = [dec.args[0].value for dec in decorators
                 if (isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "lane"
                     and dec.args and isinstance(dec.args[0], ast.Constant))]
        dump = ast.dump(node)
        if names and len(names) == len(decorators):
            for name in names:
                lanes[name] = dump
        elif names:
            # A LANE CARRYING A SECOND DECORATOR IS NOT A LANE BODY. The other
            # decorator RUNS at import and can touch anything: a fixture table,
            # a registry, another lane's defaults. Attacked 2026-09-16 with
            # `@mutate_everything` above `@lane("beta")`, which was admitted as
            # the one new lane. It goes to `others` now, so any change around
            # it answers every lane.
            others.append((getattr(node, "name", None), node, dump))
        else:
            others.append((getattr(node, "name", None), node, dump))
    return lanes, others


def _only_new_functions(old_others, new_others, old_lanes):
    """Is `new_others` `old_others` plus definitions that cannot touch what was
    already there.

    ADDITIVE IS NOT "THE DIFF HAS NO MINUS LINES". An added statement at module
    level can mutate a registry, shadow a name an existing lane resolves, or
    run a decorator. So the old statements must be present UNCHANGED and IN
    ORDER, and each addition must be an undecorated `def` whose name is new,
    whose defaults are constants (defaults are evaluated at definition time)
    and which no existing lane's code names. A class body executes when it is
    defined, so a class is not admitted."""
    old_dumps = [d for _, _, d in old_others]
    new_dumps = [d for _, _, d in new_others]
    kept = [d for d in new_dumps if d in old_dumps]
    if kept != old_dumps:
        return False
    seen = set()
    extra = []
    for name, node, dump in new_others:
        if dump in old_dumps and dump not in seen:
            seen.add(dump)
            continue
        extra.append((name, node))
    old_names = {n for n, _, _ in old_others if n}
    for name, node in extra:
        if not isinstance(node, ast.FunctionDef) or node.decorator_list or not name:
            return False
        if name in old_names or name in old_lanes:
            return False
        defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
        if any(not isinstance(d, ast.Constant) for d in defaults):
            return False
        if any(re.search(r"'%s'" % re.escape(name), dump) for dump in old_lanes.values()):
            return False
    return True


def harness_lanes(ref, path=HARNESS):
    """The lanes a diff of `tools/identity_break.py` can reach, or None when
    it can reach all of them.

    A lane commit usually edits ONE lane body in this file, or adds one. Both
    are narrowed here, and everything else returns None, which the caller reads
    as every lane: a helper, a fixture, a constant or one of the registration
    loops that build the kde, knn, radius, gp and gmm lanes can reach any
    lane at all."""
    old = _git_show(ref, path)
    if old is None:
        return None
    try:
        new = _read(path)
        old_lanes, old_others = _harness_segments(old)
        new_lanes, new_others = _harness_segments(new)
    except (OSError, SyntaxError):
        return None
    prose = set()
    if [d for _, _, d in old_others] != [d for _, _, d in new_others] \
            and not _only_new_functions(old_others, new_others, old_lanes):
        prose = _lane_keyed_prose(old_others, new_others, set(old_lanes) | set(new_lanes))
        if prose is None:
            return None                     # a shared statement moved: every lane
    return sorted(prose | {n for n in set(old_lanes) | set(new_lanes)
                           if old_lanes.get(n) != new_lanes.get(n)})


def _lane_keyed_prose(old_others, new_others, lane_names):
    """The lanes named by the keys of a LANE-KEYED TABLE OF PROSE whose
    entries were added or reworded, or None when any other shared statement
    moved.

    `NON_SIZE_REVISIONS = {"par-arima": "why ...", ...}` records, per lane,
    why a lane's revision changed. Rewording the "par-arima" sentence is a
    change to a module-level statement, which `harness_lanes` must read as
    every lane; on 2026-09-21 it was one of the two edits that sent the 0.8.12
    release to all 273. The narrowing is allowed only when ALL of these hold:
    the statements line up one to one by name and order; every one that
    differs is a plain `NAME = {...}` dict literal; the change is additions or
    changed values under unchanged keys (`_container_delta`); every added or
    changed key is a string constant naming a lane; and every added or changed
    VALUE is a string constant. A string cannot compute anything, so the most
    it can move is that lane's own record; the lanes its keys name are
    selected."""
    if [n for n, _, _ in old_others] != [n for n, _, _ in new_others] or len(old_others) != len(new_others):
        return None
    out = set()
    for (name, old, od), (_, new, nd) in zip(old_others, new_others):
        if od == nd:
            continue
        if not (isinstance(old, ast.Assign) and isinstance(new, ast.Assign) and len(new.targets) == 1
                and isinstance(new.targets[0], ast.Name) and ast.dump(old.targets[0]) == ast.dump(new.targets[0])
                and isinstance(old.value, ast.Dict) and isinstance(new.value, ast.Dict)):
            return None
        delta = _container_delta(old.value, new.value)
        if delta is None:
            return None
        added, changed = delta
        for dumped_key, value in added + changed:
            key = _key_node(old, new, dumped_key)
            if not (isinstance(key, ast.Constant) and key.value in lane_names):
                return None
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                return None
            out.add(key.value)
    return out


def pixi_toml_build_part(text):
    """pixi.toml without its task tables and comment lines: the same reading
    tools/bincache.py keys a binary on (it is copied here rather than
    imported so the selector does not depend on the cache). A task is a
    command line; dependencies, channels, environments and the workspace stay
    in."""
    out, in_tasks, in_string = [], False, False
    for line in text.splitlines():
        s = line.strip()
        if in_tasks and (s.count('"""') + s.count("'''")) % 2 == 1:
            in_string = not in_string
            continue
        if in_string:
            continue
        if s.startswith("[") and not s.startswith("[["):
            in_tasks = s.rstrip().endswith("tasks]")
            if in_tasks:
                continue
        if in_tasks or not s or s.startswith("#"):
            continue
        out.append(line.rstrip())
    return "\n".join(out) + "\n"


def pixi_tasks_only(ref, path="pixi.toml"):
    """True when pixi.toml differs from `ref` in task tables and comments
    alone. A task names a command to run later; it does not change the
    environment a binding is compiled or a lane is run in. Everything else in
    the file (a dependency, a channel, an environment, a platform) still
    selects every lane, and so does pixi.lock, which is the solved toolchain."""
    old = _git_show(ref, path)
    if old is None:
        return False
    try:
        new = _read(path)
    except OSError:
        return False
    return pixi_toml_build_part(old) == pixi_toml_build_part(new)


_MOJO_BUILD_SRC = re.compile(r"(\S+\.mojo)\b")


def build_script_roots(path, seen=None):
    """The binding sources a bindings/ build script compiles, or None when
    they cannot be read literally.

    Read from the script's own `mojo build` lines (continuations joined), the
    same reading tools/bincache.py uses to key a binary. A script that execs
    or runs ANOTHER build script by name inherits that script's roots.
    `build_host_family.sh` builds the host binding of whichever family it is
    handed, so its roots are every family's host binding. A `mojo build` line
    whose source is not a literal path returns None: every lane."""
    seen = set() if seen is None else seen
    if path in seen:
        return set()
    seen.add(path)
    try:
        text = _read(path)
    except OSError:
        return None
    if os.path.basename(path) == "build_host_family.sh":
        hs = host_surface()
        return {hs.binding_source(f["family"]) for f in hs.FAMILIES}
    roots = set()
    joined = text.replace("\\\n", " ")
    for line in joined.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        for other in re.findall(r"(?:bindings/|/)(build[\w]*\.sh)\b", s):
            rel = os.path.join("bindings", other)
            if rel != path and os.path.isfile(os.path.join(ROOT, rel)) and \
                    re.search(r"(?:^|[\s;&|(])(?:exec|sh|bash|\.)\s", s):
                more = build_script_roots(rel, seen)
                if more is None:
                    return None
                roots |= more
        if "mojo build" not in s:
            continue
        srcs = _MOJO_BUILD_SRC.findall(s)
        if len(srcs) != 1 or "$" in srcs[0]:
            return None
        src = srcs[0].strip("\"'")
        if not os.path.isfile(os.path.join(ROOT, src)):
            return None
        roots.add(src)
    return roots or None


#: NATIVE CODE THE PACKAGE LOADS BY PATH. packaging/portable_math/ builds
#: libMojolearnMath (stage.py compiles portable_math.c with its own flags) and
#: python/mojolearn/_portable_math.py loads it with ctypes. A change to any of
#: the three reaches exactly the lanes that reach the loader. The pairing is
#: CHECKED, not trusted: the loader and the builder must both still name the
#: library, or the rule is off and the path selects every lane.
NATIVE_INPUTS = {
    "packaging/portable_math/portable_math.c": "python/mojolearn/_portable_math.py",
    "packaging/portable_math/powers_of_ten.h": "python/mojolearn/_portable_math.py",
    "packaging/portable_math/stage.py": "python/mojolearn/_portable_math.py",
}
NATIVE_LIBRARY = "libMojolearnMath"


def native_input_lanes(path, rev):
    loader = NATIVE_INPUTS.get(path)
    if loader is None:
        return None
    try:
        if NATIVE_LIBRARY not in _read(loader) or \
                NATIVE_LIBRARY not in _read("packaging/portable_math/stage.py"):
            return None
    except OSError:
        return None
    return rev.get(loader, set())


def changed_paths(ref):
    """Paths that differ from `ref`, including uncommitted work: a change you
    have not committed is still a change this run has to cover.

    BOTH DIFFS, two-dot and three-dot (2026-09-21). `ref...HEAD` is what HEAD
    changed since the merge base, which misses a file the REF side changed
    when ref is not an ancestor (a release cut on its own branch, as 0.8.10 and
    0.8.11 were). `ref HEAD` is every file whose bytes differ between the two
    trees, which is the question when ref is the last verified state. For an
    ancestor the two agree; otherwise their union is the safe answer."""
    out = set()
    for args in (["diff", "--name-only", f"{ref}...HEAD"], ["diff", "--name-only", ref, "HEAD"],
                 ["diff", "--name-only", "HEAD"],
                 ["ls-files", "--others", "--exclude-standard"]):
        try:
            res = subprocess.run(["git", "-C", ROOT] + args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            # A diff that could not be taken is not an empty diff. Skipping it
            # quietly could hand the selector nothing and a pass nothing to run.
            raise SystemExit(f"REFUSING: git {' '.join(args)} failed ({exc.returncode}): "
                             f"{(exc.stderr or '').strip()[:200]}")
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
        if ref and docstring_only(ref, path):
            inert.append(path)
            reasons[path] = ("docstrings and comments only: the module's code is IDENTICAL to "
                             f"{ref} once docstrings are stripped, so it cannot move a lane body")
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
        if path in NATIVE_INPUTS:
            # BEFORE `unreachable`: stage.py is a Python file outside the
            # package that no lane imports, and it still decides the bits of a
            # library the package loads.
            native = native_input_lanes(path, rev)
            if native is not None:
                lanes |= native
                reasons[path] = (f"native code the package loads by path ({NATIVE_LIBRARY}, "
                                 f"through {NATIVE_INPUTS[path]}): {len(native)} lane(s)")
                continue
        if path == "pixi.toml" and ref and pixi_tasks_only(ref, path):
            inert.append(path)
            reasons[path] = ("pixi.toml task tables and comments only: the environment, "
                             "dependencies and channels are identical to " + ref)
            continue
        if path in HARNESS_RUNTIME_IMPORTS:
            # BEFORE the test-module and unreachable rules: nothing a lane
            # reaches names it, and the harness still runs it on every column.
            fallback = True
            unattributed.append(path)
            reasons[path] = ("imported by the harness while it records a column, and what it "
                             "writes there (degenerate_lanes) is derived over every lane: every lane")
            continue
        why_test = test_module_inert(path)
        if why_test:
            inert.append(path)
            reasons[path] = why_test
            continue
        why_unreachable = unreachable(path)
        if why_unreachable:
            inert.append(path)
            reasons[path] = why_unreachable
            continue
        if path in SELECTION_MACHINERY:
            inert.append(path)
            reasons[path] = ("the selection machinery itself: it decides which lanes run and "
                             "cannot move a lane's bits (tools/test_lane_select.py covers it)")
            continue
        if path in GLOBAL_PATHS or path in enumerator_files():
            if not ref:
                fallback = True
                unattributed.append(path)
                reasons[path] = ("a registry of the whole binding surface. Whether this change is "
                                 "an ADDITION can only be read from a diff, and no ref was given, "
                                 "so: every lane. Use --changed-since to get the narrow answer")
                continue
            added = registry_lanes(ref, path, sources)
            if added is not None:
                lanes |= set(added)
                reasons[path] = (f"a whole-surface registry, but the diff only ADDS to it: every "
                                 f"existing statement is present and in order, and what was added "
                                 f"names {len(added)} lane(s)")
                continue
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
        if path.startswith("bindings" + os.sep) and path.endswith(".sh"):
            roots = build_script_roots(path)
            if roots is not None:
                built = set().union(*[rev.get(r, set()) for r in roots])
                lanes |= built
                reasons[path] = (f"a build script: it compiles {len(roots)} binding source(s) "
                                 f"({','.join(sorted(roots)[:3])}{'...' if len(roots) > 3 else ''}), "
                                 f"which {len(built)} lane(s) reach")
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


def census(limit):
    """The files the map credits with FEW lanes, which is where a missing edge
    hides. A file serving six lanes that the map credits with three is the
    shape to look for; that is exactly what `neural_inference.py` was on
    2026-09-16, credited with mlp and the two transformer lanes while also
    serving four mamba and two samba lanes through subclasses.

    The two inversion checks in `tools/test_lane_select.py` are the
    mechanical half of this and run over the whole tree. This is the half a
    person reads."""
    sources, _ = lane_sources()
    rev = reverse_map(sources)
    rows = sorted(((len(lanes), rel) for rel, lanes in rev.items() if len(lanes) <= limit),
                  key=lambda r: (r[0], r[1]))
    print(f"# {len(rows)} file(s) attributed to {limit} lane(s) or fewer, of {len(rev)} mapped")
    for n, rel in rows:
        defines = ""
        if rel.endswith(".py"):
            tree = _parse(rel)
            names = [x.name for x in (tree.body if tree else [])
                     if isinstance(x, (ast.ClassDef, ast.FunctionDef))]
            defines = " defines " + ",".join(names[:6]) + ("..." if len(names) > 6 else "")
        print(f"#   {n:>3} {rel}{defines}")
    print(f"# {len(rows)} file(s); a file here that serves more lanes than this is a missing edge")
    return 0


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
    ap.add_argument("--census", type=int, default=0, metavar="N",
                    help="list the files the map attributes to N lanes or fewer, with what each "
                         "file defines; a missing edge hides in a file credited with too few")
    args = ap.parse_args(argv)

    if args.count:
        print(len(all_lanes()))
        return 0
    if args.selfcheck:
        return selfcheck()
    if args.census:
        return census(args.census)

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
