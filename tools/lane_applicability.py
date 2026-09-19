#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHICH LANES ARE MEANINGFUL ON WHICH COLUMN, and what each one is compared
against (lane/oracle-and-applicability-audit, 2026-09-16).

`tools/lane_select.py` answers a DIFFERENT question, "which lanes can a change
possibly move". This one answers "on which columns can a lane's proposition
even be stated", and "when this cell passes, what did it pass against".

WHY IT EXISTS. `identity_break.lane` is three lines that put a function in a
dict. A lane declares no column, no vendor, no device count and no tier, so
"run every lane on every column" was never a decision anyone made: it is the
only thing the structure can express. The set is then trimmed by hand at each
call site, and a hand-kept trim rots into agreeing with everything.

Two consequences this file measures rather than asserts:

  A DEGENERATE CELL is a cell whose PREMISE cannot hold on the column it runs
  on. The `par-*` driver lanes are the large case: `_par_devices()` defaults
  to device "0", and the drivers' whole claim, in their own docstring, is that
  a two-device column hashes equal cell for cell to the one-device column. On
  a box with one GPU that claim is not false, it is not expressible, so the
  cell exercises a sharding driver with one shard and cannot fail for the
  reason the lane exists. The same shape appears for a lane whose arithmetic
  is the CPU host route: on an Apple or NVIDIA column it measures that box's
  CPU and says nothing about Metal or CUDA. Its limit is the PURE-PYTHON lane,
  which stands on no Mojo binding at all: `cross-val-folds` hashes integer
  fold bookkeeping that no vendor can move, so five of the six columns are
  vacuous and the sixth, cpu-host, is the only one where it says anything.
  DEGENERACY CUTS BOTH WAYS, and until 2026-09-19 this file only cut one: a
  lane with no binding was refused on cpu-host for "no CPU route" (a rule
  written for GPU-only lanes) and admitted on all five GPU columns, so the one
  meaningful cell was the one that never ran.

  THE ORACLE is what a passing cell was compared against. Three classes, and
  they are not equally strong:
    independent  a DIFFERENT implementation of the same thing, inside the
                 cell, raising when they disagree: `par-forest` holds the
                 sharded driver against a plain `RandomForestClassifier` fit.
                 This one can fail on its own, on one box, with no record.
    self         the same code asked twice, or asked through two doors of the
                 same fitted object: `kmeans` holding `predict(X)` to
                 `labels_`, the forecasters holding `forecast` to `predict`,
                 and the harness's own double fit. It proves run-to-run and
                 door-to-door stability. It cannot detect a defect that both
                 sides share, which is every defect in the arithmetic.
    recorded     nothing inside the run. The cell's value is a hash, and it
                 only means something when `--diff` holds it against a
                 previously recorded hash of THE SAME CODE. A defect present
                 when the reference was recorded is invisible forever.

DERIVED, NOT HAND-WRITTEN, for the reason `lane_select.py` gives at length: a
hand-kept list agrees with everything a few weeks after it is written, which
is the shape of six failures in this repository. Every fact here comes from
something else already enforces:

  the registry   `identity_break.LANES`, read by IMPORT, not by grep, so the
                 23 lanes that register by call and not by decorator are in.
  the lane body  `inspect.getsource` of the lane function, parsed. The device
                 requirement is `_par_devices` appearing in the body's
                 transitive closure over the harness's own helpers, never the
                 `par-` prefix; the prefix is then CHECKED against the derived
                 answer and a disagreement is an error, so a driver lane that
                 stops calling the helper, or a non-driver lane that starts,
                 is caught rather than assumed.
  the oracle     every in-cell comparison that RAISES (`_same_bytes`), with
                 its two sides resolved to their root bindings. Two sides
                 rooted at two separately constructed objects is an
                 independent oracle; two sides rooted at the same object is
                 self-comparison. A lane with no such call is `recorded`.
  the CPU route  `python/mojolearn/host_surface.py`'s own covered-lane
                 answer, and the binding names each public class's door file
                 mentions: a class whose door names only `*_host` bindings
                 computes on the CPU wherever it runs.
  no binding     the same two sources read for ABSENCE: no host family
                 declares the lane, and no package file behind the doors it
                 opens names a `_mojolearn_*` binding. host_surface states the
                 same fact in `PUBLIC_HOST_ONLY_LANES` (a family name, or None
                 for "no native host binding is required"), and the two are
                 CHECKED against each other exactly as the device claim is
                 checked against the `par-` prefix.
  the record     `identity_break.record_excluded_lanes()`, the harness's own
                 statement of what a release record does not cover.

REFUSAL, NOT A QUIET SKIP. `check()` raises `LaneNotApplicable` and names
every degenerate cell with its reason. A mechanism that can only skip is the
hand-kept list in a different shirt: a skipped cell and a passing cell look
the same in a summary line, and this repository has already shipped two
shrunken cells that went blind and read as passing.

    python3 tools/lane_applicability.py --table
    python3 tools/lane_applicability.py --column apple-metal --degenerate
    python3 tools/lane_applicability.py --oracle-counts
    python3 tools/lane_applicability.py --check --column apple-metal --lanes par-forest
    python3 tools/lane_applicability.py --selfcheck
"""
import argparse
import ast
import collections
import importlib.util
import inspect
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "python", "mojolearn")
HARNESS = os.path.join(ROOT, "tools", "identity_break.py")
MANIFEST = os.path.join(PKG, "host_surface.py")

#: The fixtures a full cell sweep runs, from the harness itself.
#: A "cell" in this file is one lane on one fixture on one column.


class LaneNotApplicable(Exception):
    """A lane was asked to run on a column its own premise cannot hold on."""


# ------------------------------------------------------------------ columns
class Column:
    """A place a cell can run. `devices` is how many GPUs the column offers
    the drivers; `route` is which arithmetic the public surface takes there."""

    def __init__(self, name, vendor, backend, devices, route, note=""):
        self.name, self.vendor, self.backend = name, vendor, backend
        self.devices, self.route, self.note = devices, route, note

    def __repr__(self):
        return f"Column({self.name}, {self.devices} device(s), {self.route})"


COLUMNS = {c.name: c for c in (
    Column("apple-metal", "apple", "metal", 1, "gpu",
           "one Mac, one GPU, one Metal job at a time, cannot be rented or parallelized"),
    Column("nvidia-1gpu", "nvidia", "cuda", 1, "gpu", "a rented single-GPU pod"),
    Column("nvidia-2gpu", "nvidia", "cuda", 2, "gpu", "the two-device leg"),
    Column("amd-1gpu", "amd", "hip", 1, "gpu", "a rented single-GPU pod"),
    Column("amd-2gpu", "amd", "hip", 2, "gpu", "the two-device leg"),
    Column("cpu-host", "cpu", "host", 0, "host",
           "the rented CPU pod and the M4 CPU: the host route, no device kernel"),
)}


# ------------------------------------------------------------------ loading
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


_IB = None
_HS = None


def use_harness(mod):
    """Use an ALREADY-LOADED harness instead of executing the file again.

    `identity_break.py` calls this on itself so it can ask whether the lanes
    it is about to RECORD can state their proposition on the column it is
    recording (2026-09-19). Without it, importing this module from inside the
    harness would exec ~10k lines a second time under a second name and
    re-register all 247 lanes, and the two registries could then drift inside
    one process -- which is the exact class of bug this file exists to catch.
    """
    global _IB
    _IB = mod


def identity_break():
    """The harness, IMPORTED. The registry is the source of the lane set; no
    grep of `@lane(` sees the 23 lanes that register by call."""
    global _IB
    if _IB is None:
        if os.path.join(ROOT, "python") not in sys.path:
            sys.path.insert(0, os.path.join(ROOT, "python"))
        _IB = _load("_applicability_identity_break", HARNESS)
    return _IB


def host_surface():
    global _HS
    if _HS is None:
        _HS = _load("_applicability_host_surface", MANIFEST)
    return _HS


# ------------------------------------------------------- the harness's AST
_HARNESS_AST = None


def _harness_functions():
    """Every module-level `def` in the harness, by name, as an AST node. The
    lane bodies' helper closure is walked over these."""
    global _HARNESS_AST
    if _HARNESS_AST is None:
        tree = ast.parse(open(HARNESS, encoding="utf-8").read())
        out = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.setdefault(node.name, []).append(node)
        _HARNESS_AST = out
    return _HARNESS_AST


def _lane_source(fn):
    """The lane function's own source. The lanes registered by call are
    closures returned by a factory, and `getsource` gives the inner `def`,
    which is the body that runs."""
    return inspect.getsource(fn)


def _lane_tree(fn):
    src = inspect.cleandoc("\n".join(_lane_source(fn).splitlines()))
    # A decorated lane's source starts at the decorator; dedent by parsing the
    # function node out of a module parse of the dedented text.
    lines = _lane_source(fn).splitlines()
    indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    src = "\n".join(l[indent:] if len(l) >= indent else l for l in lines)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    raise ValueError("no function node in lane source")


def _names_in(node):
    """Every bare name and attribute name the node mentions."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
        elif isinstance(sub, ast.ImportFrom):
            for a in sub.names:
                out.add(a.name)
            if sub.module:
                out.add(sub.module.split(".")[-1])
    return out


def _called(node):
    """The harness helpers a body CALLS BY BARE NAME: `_par_devices()`,
    `_same_bytes(...)`, `_fit(...)`, plus anything it imports by name.

    Deliberately narrower than `_names_in`. Walking every Name and attribute
    made `rf-clf` reach `_par_devices`, because `fit` and `predict` are also
    the names of harness helpers and of estimator methods, and one collision
    is enough to hand every lane every helper. lane_select pays for the same
    collision with its ENUMERATOR_MAX_BINDINGS sink. Here the fix is the call
    graph itself: a bare-name call is unambiguous, an attribute is not."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            out.add(sub.func.id)
        elif isinstance(sub, ast.ImportFrom):
            for a in sub.names:
                out.add(a.name)
    return out


def _closure(node, limit=40):
    """The harness helper names the lane body reaches through calls,
    transitively. The same shape as `lane_select`'s import walk, over `def`s
    instead of files."""
    funcs = _harness_functions()
    frontier = _called(node)
    reached, seen, depth = set(frontier), set(), 0
    while frontier and depth < limit:
        nxt = set()
        for name in frontier:
            if name in seen or name not in funcs:
                continue
            seen.add(name)
            for f in funcs[name]:
                for n in _called(f):
                    if n not in reached:
                        reached.add(n)
                        nxt.add(n)
        frontier, depth = nxt, depth + 1
    return reached


def _ml_attrs(node):
    """The public classes and functions the lane asks `ml` for: `ml.KMeans`,
    `ml.linalg`. This is the lane's door onto the package."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "ml":
            out.add(sub.attr)
    return out


# ------------------------------------------------------------- the oracle
def _root(node):
    """The root binding an expression hangs off: `par.predict_proba(X)` is
    rooted at `par`, `np.asarray(e.labels_)` at `np`... so unwrap the common
    numpy wrappers first and take the innermost non-numpy Name."""
    wrappers = {"np", "npx"}
    cur = node
    guard = 0
    while guard < 64:
        guard += 1
        if isinstance(cur, ast.Call):
            root = _root_name(cur.func)
            if root in wrappers and cur.args:
                cur = cur.args[0]
                continue
            return root
        if isinstance(cur, (ast.Attribute, ast.Name)):
            return _root_name(cur)
        if isinstance(cur, ast.Subscript):
            cur = cur.value
            continue
        if isinstance(cur, ast.Tuple) and cur.elts:
            cur = cur.elts[0]
            continue
        return None
    return None


def _root_name(node):
    cur = node
    while isinstance(cur, (ast.Attribute, ast.Subscript, ast.Call)):
        cur = cur.func if isinstance(cur, ast.Call) else cur.value
    return cur.id if isinstance(cur, ast.Name) else None


#: The in-cell comparisons that RAISE. A comparison that does not raise is not
#: an oracle: it is a value someone may or may not look at.
RAISING_COMPARE = ("_same_bytes",)


def _oracle(node):
    """The lane's in-cell oracle class and the evidence for it.

    A `_same_bytes` call whose two sides root at two DIFFERENT bindings is an
    independent oracle: two objects were built by two call chains and held to
    each other. Rooted at the SAME binding it is self-comparison: one fitted
    object asked through two of its own doors, which proves the doors agree
    and cannot see a defect they share. No such call at all is `recorded`:
    the cell is a hash and means nothing until `--diff` holds it against a
    previous run of the same code."""
    pairs = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and _root_name(sub.func) in RAISING_COMPARE:
            args = [a for a in sub.args]
            if len(args) >= 4:
                a, b = _root(args[1]), _root(args[3])
            elif len(args) == 2:
                a, b = _root(args[0]), _root(args[1])
            else:
                continue
            pairs.append((a, b, _literal(args[0]), _literal(args[2]) if len(args) >= 3 else ""))
    if not pairs:
        return "recorded", []
    if any(a is not None and b is not None and a != b for a, b, _, _ in pairs):
        return "independent", pairs
    return "self", pairs


def _literal(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else ""


# -------------------------------------------------------- the CPU host route
_DOOR_BINDINGS = None


def _door_bindings():
    """public name -> the `_mojolearn_*` bindings the package file defining it
    mentions. A name whose bindings are ALL `*_host` computes on the CPU
    wherever the column runs."""
    global _DOOR_BINDINGS
    if _DOOR_BINDINGS is not None:
        return _DOOR_BINDINGS
    binding_re = re.compile(r"_mojolearn[a-z0-9_]*")
    out = {}
    for fn in sorted(os.listdir(PKG)):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(PKG, fn)
        try:
            text = open(path, encoding="utf-8").read()
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        bindings = set(binding_re.findall(text))
        if not bindings or len(bindings) > 3:
            # A file naming more than three bindings is a REGISTRY, not a
            # door (`_backend.py`, `host_surface.py`). lane_select measured
            # the same threshold and for the same reason: harvesting from a
            # registry gives every name every binding.
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                out.setdefault(node.name, set()).update(bindings)
    _DOOR_BINDINGS = out
    return out


def _host_only_names(ml_attrs):
    """The lane's public names whose only binding is a host binding."""
    doors = _door_bindings()
    out = set()
    for name in ml_attrs:
        b = doors.get(name)
        if b and all(x.endswith("_host") for x in b):
            out.add(name)
    return out


_PACKAGE_BINDINGS = None


def _package_bindings():
    """(module stem -> the `_mojolearn_*` bindings its file names, public name
    -> the module stems that define it), over the WHOLE package.

    `_door_bindings` above answers "which bindings is this name's door built
    on", and drops a file naming more than three because harvesting from a
    registry gives every name every binding. This pair answers the opposite
    question, "does this file name a binding AT ALL", where a registry's many
    bindings are a `yes` and that threshold would throw the answer away."""
    global _PACKAGE_BINDINGS
    if _PACKAGE_BINDINGS is not None:
        return _PACKAGE_BINDINGS
    binding_re = re.compile(r"_mojolearn[a-z0-9_]*")
    modules, names = {}, {}
    for fn in sorted(os.listdir(PKG)):
        if not fn.endswith(".py"):
            continue
        try:
            text = open(os.path.join(PKG, fn), encoding="utf-8").read()
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        stem = fn[:-3]
        modules[stem] = set(binding_re.findall(text))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                names.setdefault(node.name, set()).add(stem)
    _PACKAGE_BINDINGS = (modules, names)
    return _PACKAGE_BINDINGS


def _pure_python(ml_attrs, has_cpu_route):
    """Whether the lane's arithmetic stands on NO Mojo binding of any kind.

    THE HOLE THIS CLOSES (2026-09-19). `cross-val-folds` hashes
    `model_selection._default_folds`, integer index bookkeeping in Python, and
    `python/mojolearn/model_selection.py` names no binding at all. No host
    FAMILY can declare a lane it builds no binding for, so `covered_lanes()`
    does not hold it, and the rule below read that absence as "the CPU column
    has no arithmetic to run and the cell REFUSES". It does not refuse: it is
    the one lane that answers on a CPU-only wheel with no binding installed,
    which is its own docstring's claim, and the sabotage record at
    `2026-09-17_sabotage-sweep/e-python-lanes/cpu-x86.sabotage.json` is a CPU
    cell that ran. So the rule refused it on the ONE column where its
    proposition is meaningful and admitted it on five columns where no GPU
    code runs at all, which is backwards in both directions at once.

    A lane with no binding anywhere is not degenerate on the CPU column; it is
    a PURE-PYTHON lane, meaningful on exactly the host column and vacuous on
    every other, which is the same shape as `host_only` one step further out.

    DERIVED, never a list of names, and from two facts that must BOTH hold:

      no family   `host_surface.covered_lanes()` does not hold it, so nothing
                  in the tree builds a binding for its arithmetic.
      no binding  every public name the lane opens resolves to a package file,
                  and not one of those files names a `_mojolearn_*` binding.

    NEITHER ALONE IS RIGHT, measured on this registry. Six lanes pass the
    second and fail the first -- `mlp`, `par-mlp`, `optim-sgd`,
    `optim-adam-clip`, `training-primitives`, `cross-entropy-arms` -- because
    `training.py` and `_mlp_impl.py` are re-export shims over
    `_training_impl`, which is where the bindings are named; a file-level scan
    that stopped there would have called six binding-backed lanes pure Python.
    And the 21 factory lanes that open no `ml.<name>` at all pass the second
    vacuously: an empty door set is no evidence, so it answers False.
    """
    if has_cpu_route or not ml_attrs:
        return False
    modules, names = _package_bindings()
    reached = set()
    for attr in ml_attrs:
        if attr in modules:
            reached.add(attr)
        elif attr in names:
            reached |= names[attr]
        else:
            # A door this file cannot resolve to a package file is not
            # evidence of absence; the lane keeps whatever it had.
            return False
    return not any(modules[stem] for stem in reached)


_COVERED = None


def covered_lanes():
    """The lanes host_surface says have a CPU route at all, from its own
    answer, never a copy of it."""
    global _COVERED
    if _COVERED is None:
        hs = host_surface()
        _COVERED = set(hs.covered_lanes())
    return _COVERED


# ------------------------------------------------------------------ scope
class Scope:
    """What a lane needs for its proposition to be expressible, derived."""

    def __init__(self, name, claim_devices, oracle, oracle_evidence,
                 host_only, has_cpu_route, is_function, record_excluded, ml_attrs,
                 pure_python=False):
        self.name = name
        self.claim_devices = claim_devices
        self.oracle = oracle
        self.oracle_evidence = oracle_evidence
        self.host_only = host_only
        self.has_cpu_route = has_cpu_route
        self.pure_python = pure_python
        self.is_function = is_function
        self.record_excluded = record_excluded
        self.ml_attrs = ml_attrs

    @property
    def anchor(self):
        """WHAT HOLDS THIS LANE DOWN, strongest first. The three classes the
        audit asked for, plus the one the tree actually leans on:

          in-cell-independent  a different implementation, in the cell,
                               raising on disagreement. Fails on one box with
                               no record.
          in-cell-self         one fitted object through two of its own
                               doors. Fails on one box, but only for a defect
                               the two doors do not share.
          cross-route          nothing in the cell, but the lane has BOTH a
                               device kernel and a CPU host route, which are
                               two separate implementations in the tree, so
                               `--diff` of a GPU column against the CPU
                               column is an independent comparison. It proves
                               nothing from ONE column, which is the point:
                               running this lane twice on the same column
                               adds no evidence at all.
          recorded-only        neither. The cell is a hash whose only
                               comparison is a previous hash of the same
                               code, so a defect present when the reference
                               was recorded is invisible forever."""
        if self.oracle == "independent":
            return "in-cell-independent"
        if self.oracle == "self":
            return "in-cell-self"
        return "cross-route" if self.has_cpu_route else "recorded-only"

    @property
    def kind(self):
        if self.claim_devices >= 2:
            return "multi-device-driver"
        if self.pure_python:
            return "pure-python-host"
        if self.host_only:
            return "cpu-host-route-only"
        if self.is_function:
            return "vendor-independent-arithmetic"
        if not self.has_cpu_route:
            return "gpu-only"
        return "estimator-both-routes"

    def applicable(self, column):
        """(ok, reason). `ok` False means the lane's PREMISE cannot hold on
        this column, which is not the same as the lane failing there."""
        col = COLUMNS[column] if isinstance(column, str) else column
        if self.claim_devices >= 2 and col.devices < 2:
            rest = ("the in-cell comparison against a plain fit still fires, so the LOGICAL SHARD "
                    "split is tested here; the DEVICE axis is not"
                    if self.oracle == "independent" else
                    "and this lane asserts nothing in the cell either, so on this column it is a hash "
                    "of one shard on one device held only against a previous hash of itself")
            return False, (
                f"DEGENERATE (device axis): a multi-device driver lane on a column with "
                f"{col.devices} device(s). Its claim, in `_par_devices`'s own docstring, is that a "
                f"two-device column hashes equal cell for cell to the one-device column; with one "
                f"shard that equality is not false, it is not expressible. {rest}")
        if self.pure_python and col.route != "host":
            return False, (
                f"DEGENERATE (vacuous): the lane stands on no Mojo binding of any kind "
                f"({', '.join(self.ml_attrs)} names none), so on the {col.name} column it runs the "
                f"same Python on that box's CPU and says nothing whatever about {col.backend}. Its "
                f"proposition is stateable on the cpu-host column and on no other")
        if self.host_only and col.route != "host":
            return False, (
                f"DEGENERATE: the lane's arithmetic is the CPU host route ({', '.join(sorted(self.host_only))}), "
                f"so on the {col.name} column it measures that box's CPU and says nothing about "
                f"{col.backend}")
        if not self.has_cpu_route and not self.pure_python and col.route == "host":
            return False, (
                "DEGENERATE: host_surface declares no CPU route for this lane, so the CPU column "
                "has no arithmetic to run and the cell REFUSES rather than answering. A refusal and "
                "a pass read the same in a column total, which is how a CPU column reported "
                "coverage it did not have")
        return True, "applicable"


_SCOPES = None


def scopes():
    """lane -> Scope for every registered lane. Derived at call time."""
    global _SCOPES
    if _SCOPES is not None:
        return _SCOPES
    ib = identity_break()
    hs = host_surface()
    excluded = set(ib.record_excluded_lanes())
    covered = covered_lanes()
    out = {}
    disagreements = []
    for name, fn in ib.LANES.items():
        node = _lane_tree(fn)
        reached = _closure(node)
        ml_attrs = _ml_attrs(node)
        oracle, evidence = _oracle(node)
        claim_devices = 2 if "_par_devices" in reached else 1
        # THE DERIVED ANSWER IS CHECKED AGAINST THE NAME, never taken from it.
        # A `par-` lane that stopped calling the helper, or a lane that started,
        # is a disagreement and is reported, not silently resolved.
        if (name.startswith("par-")) != (claim_devices >= 2):
            disagreements.append(
                f"{name}: name says {'driver' if name.startswith('par-') else 'not a driver'}, "
                f"body says {'reaches' if claim_devices >= 2 else 'does not reach'} _par_devices")
        out[name] = Scope(
            name=name,
            claim_devices=claim_devices,
            oracle=oracle,
            oracle_evidence=evidence,
            host_only=_host_only_names(ml_attrs),
            has_cpu_route=name in covered,
            is_function=_is_function_lane(node),
            record_excluded=name in excluded,
            ml_attrs=sorted(ml_attrs),
            pure_python=_pure_python(ml_attrs, name in covered),
        )
    # THE DERIVED PURE-PYTHON ANSWER IS CHECKED AGAINST host_surface's OWN
    # DECLARATION, the same way the device claim is checked against the lane
    # name. `PUBLIC_HOST_ONLY_LANES` maps a host-only lane to the family whose
    # binding it needs, or to None for "pure Python: no native host binding is
    # required". That is the package stating the fact; this file deriving it
    # from the doors is a second witness, and the two disagreeing is a defect
    # in one of them, never something to resolve quietly here.
    declared = {n for n, fam in getattr(hs, "PUBLIC_HOST_ONLY_LANES", {}).items() if fam is None}
    derived = {n for n, sc in out.items() if sc.pure_python}
    for name in sorted(declared - derived):
        disagreements.append(
            f"{name}: host_surface.PUBLIC_HOST_ONLY_LANES declares it pure Python (no host "
            f"binding required); the doors say otherwise"
            + (" (it is not a registered lane)" if name not in out else ""))
    for name in sorted(derived - declared):
        disagreements.append(
            f"{name}: derived pure Python (no host family declares it and no package file behind "
            f"its doors names a binding); host_surface.PUBLIC_HOST_ONLY_LANES does not declare it")
    _SCOPES = out
    _SCOPES_DISAGREE[:] = disagreements
    return out


_SCOPES_DISAGREE = []


def _is_function_lane(node):
    """A lane that hashes a function's output and returns no estimator: every
    `_fit(...)` call in it passes one argument. `Fit.probe` then stays at its
    class default `n/a:function` and no infer, model or batch cell exists."""
    calls = [c for c in ast.walk(node) if isinstance(c, ast.Call) and _root_name(c.func) == "_fit"]
    return bool(calls) and all(len(c.args) == 1 and not c.keywords for c in calls)


# ------------------------------------------------------------------ refusal
def check(lanes, column, allow=()):
    """RAISE unless every lane's premise holds on `column`. `allow` names
    lanes the caller has decided to run anyway, and each one is still printed,
    so an exception is a written-down decision and not a silent skip."""
    s = scopes()
    bad = []
    for name in lanes:
        if name not in s:
            raise LaneNotApplicable(f"REFUSING: {name!r} is not a registered lane")
        ok, why = s[name].applicable(column)
        if not ok and name not in allow:
            bad.append(f"  {name}: {why}")
    if bad:
        raise LaneNotApplicable(
            f"REFUSING: {len(bad)} of {len(lanes)} lane(s) cannot state their proposition on the "
            f"{column} column. Running them there produces cells that pass because nothing they "
            f"assert can fail, which reads in a summary exactly like coverage.\n" + "\n".join(bad) +
            "\n  Drop them from this column, or name them in `allow` with a reason.")
    return True


def degenerate(column):
    """lane -> reason, for every lane degenerate on `column`."""
    return {n: s.applicable(column)[1] for n, s in scopes().items() if not s.applicable(column)[0]}


# ------------------------------------------------------------------- report
def _fixtures():
    return list(identity_break().FIXTURES)


def table():
    s = scopes()
    rows = []
    for name in sorted(s):
        sc = s[name]
        cols = {c: ("yes" if sc.applicable(c)[0] else "DEGENERATE") for c in COLUMNS}
        rows.append(dict(lane=name, kind=sc.kind, oracle=sc.oracle, anchor=sc.anchor,
                         claim_devices=sc.claim_devices,
                         cpu_route=sc.has_cpu_route, record=not sc.record_excluded,
                         columns=cols))
    return rows


def _selfcheck():
    """The mechanism must REFUSE, and must be watched refusing. Every arm here
    is run on the side where it should FAIL first."""
    s = scopes()
    fails = []

    def want(cond, msg):
        if not cond:
            fails.append(msg)

    want(len(s) == len(identity_break().LANES), "scopes() lost lanes")
    want(not _SCOPES_DISAGREE,
         "a derived answer disagrees with the declaration it is checked against: "
         + "; ".join(_SCOPES_DISAGREE))

    # 1. THE REFUSAL FIRES. A driver lane on a one-device column must raise.
    try:
        check(["par-forest"], "apple-metal")
        fails.append("check() did NOT refuse par-forest on a one-device column")
    except LaneNotApplicable as exc:
        want("par-forest" in str(exc) and "DEGENERATE" in str(exc),
             "the refusal did not name the lane and the reason")

    # 2. THE REFUSAL IS NOT UNIVERSAL. The same lane on a two-device column
    #    must pass, or the refusal is a blanket and proves nothing.
    check(["par-forest"], "nvidia-2gpu")

    # 3. A PLAIN LANE IS NOT REFUSED on a GPU column.
    check(["rf-clf"], "apple-metal")

    # 4. AN UNKNOWN LANE IS REFUSED, not skipped.
    try:
        check(["no-such-lane"], "apple-metal")
        fails.append("check() accepted an unregistered lane")
    except LaneNotApplicable:
        pass

    # 5. THE ORACLE CLASSES ARE DISTINGUISHED, on lanes read by hand from the
    #    source, so a classifier that answered one class for everything fails.
    want(s["par-forest"].oracle == "independent",
         "par-forest is a plain fit held against a sharded fit; it must read independent")
    want(s["rf-clf"].oracle == "recorded",
         "rf-clf asserts nothing in the cell; it must read recorded")
    self_lanes = [n for n in s if s[n].oracle == "self"]
    want(self_lanes, "no lane classified self: the classifier collapsed")
    want(len(set(s[n].oracle for n in s)) == 3, "fewer than three oracle classes appear")
    want(s["holtwinters"].oracle == "self",
         "holtwinters holds `forecast` to `predict` on ONE fitted object; it must read self")
    want(len(set(s[n].anchor for n in s)) == 4, "fewer than four anchor classes appear")
    want(s["gp-optimize"].anchor == "cross-route",
         "gp-optimize has a CPU host gradient route; it must read cross-route")
    want(s["kmeans"].anchor == "cross-route",
         "kmeans has a CPU host route and no in-cell oracle; it must read cross-route")

    # 6b. THE PURE-PYTHON LANE IS MEANINGFUL ON THE CPU COLUMN AND VACUOUS ON
    #     THE GPU ONES (2026-09-19). Until this arm the rule held exactly the
    #     opposite for `cross-val-folds`: refused on cpu-host as DEGENERATE
    #     ("host_surface declares no CPU route"), admitted on all five GPU
    #     columns where no GPU code runs at all. Both halves are asserted here,
    #     because fixing only the refusal would leave five vacuous cells.
    want(s["cross-val-folds"].pure_python,
         "cross-val-folds stands on no binding at all; it must read pure_python")
    try:
        check(["cross-val-folds"], "cpu-host")
    except LaneNotApplicable as exc:
        fails.append("check() REFUSED the pure-Python lane on the one column its "
                     "proposition is stateable on: " + str(exc).splitlines()[1].strip())
    try:
        check(["cross-val-folds"], "apple-metal")
        fails.append("check() did NOT refuse the pure-Python lane on a GPU column")
    except LaneNotApplicable as exc:
        want("vacuous" in str(exc) and "metal" in str(exc),
             "the GPU refusal did not say the cell is vacuous on that backend")

    # 6c. THE CPU EXEMPTION IS NARROW, watched on BOTH sides of the one branch
    #     this change touched. The exemption is for "stands on no binding
    #     anywhere", not for "host_surface does not cover it", and the second
    #     must still be refused.
    #
    #     ON A CONSTRUCTED SCOPE, and deliberately. Measured 2026-09-19: every
    #     registered lane that `covered_lanes()` misses is either one of the 32
    #     `par-*` drivers, where the DEVICE rule above fires first and this
    #     branch is never reached, or `cross-val-folds` itself. So no lane in
    #     the registry can watch this branch refuse, and an arm that picked one
    #     would be watching the device rule under another name. The rule is the
    #     real `Scope.applicable`; only the inputs are made up.
    backed = Scope(name="<constructed>", claim_devices=1, oracle="recorded", oracle_evidence=[],
                   host_only=set(), has_cpu_route=False, is_function=True, record_excluded=True,
                   ml_attrs=["Constructed"], pure_python=False)
    ok, why = backed.applicable("cpu-host")
    want(not ok and "no CPU route" in why,
         "an uncovered binding-backed lane must still be REFUSED on the CPU column: " + why)
    backed.pure_python = True
    ok, why = backed.applicable("cpu-host")
    want(ok, "a pure-Python lane must RUN on the CPU column, not refuse: " + why)
    ok, why = backed.applicable("nvidia-1gpu")
    want(not ok and "vacuous" in why,
         "a pure-Python lane must be degenerate on a GPU column: " + why)

    # 6d. THE PURE-PYTHON ANSWER IS NOT host_surface's DECLARATION REPEATED.
    #     The four tokenizer lanes are host-only AND declared in the same dict,
    #     with a family name rather than None; they stand on a binding and must
    #     not read pure. `mlp` and `training-primitives` open doors whose own
    #     file names no binding (`training.py` and `_mlp_impl.py` are shims
    #     over `_training_impl`) and must not read pure either.
    for n in ("tokenizer", "bpe-trainer", "bpe-vocabulary", "tokenized-corpus",
              "mlp", "par-mlp", "optim-sgd", "training-primitives", "cross-entropy-arms"):
        want(not s[n].pure_python, f"{n} stands on a binding; it must not read pure_python")
    pure = sorted(n for n, sc in s.items() if sc.pure_python)
    want(pure == ["cross-val-folds"],
         "the pure-Python set must be exactly cross-val-folds on this registry; it is "
         + (", ".join(pure) if pure else "empty"))

    # 6. THE DERIVATION IS NOT READING THE LANE NAMES. Renaming a driver lane
    #    must not change its answer, and a lane renamed TO `par-` must not
    #    acquire the two-device claim.
    want(la_scope_of_body(identity_break().LANES["par-kmeans"]) == 2,
         "the device claim did not come from the body")
    want(la_scope_of_body(identity_break().LANES["kmeans"]) == 1,
         "a plain lane picked up the two-device claim")
    return fails


def la_scope_of_body(fn):
    """The device claim read from a function OBJECT alone, with no name in
    hand. The selfcheck uses it to prove the derivation is not the prefix."""
    return 2 if "_par_devices" in _closure(_lane_tree(fn)) else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--table", action="store_true", help="the full truth table, TSV")
    p.add_argument("--json", action="store_true", help="the truth table as JSON")
    p.add_argument("--column", default="apple-metal", choices=sorted(COLUMNS))
    p.add_argument("--degenerate", action="store_true", help="the degenerate lanes on --column")
    p.add_argument("--counts", action="store_true", help="the per-column degenerate counts")
    p.add_argument("--oracle-counts", action="store_true")
    p.add_argument("--check", action="store_true", help="refuse degenerate lanes on --column")
    p.add_argument("--lanes", nargs="+", default=None)
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args(argv)

    if a.selfcheck:
        fails = _selfcheck()
        for f in fails:
            print("SELFCHECK FAIL:", f)
        print("selfcheck:", "FAILED" if fails else "ok")
        return 1 if fails else 0

    s = scopes()
    if _SCOPES_DISAGREE:
        for d in _SCOPES_DISAGREE:
            print("DERIVATION DISAGREES WITH THE NAME:", d, file=sys.stderr)

    if a.check:
        lanes = a.lanes or sorted(s)
        try:
            check(lanes, a.column)
        except LaneNotApplicable as exc:
            print(exc)
            return 2
        print(f"ok: {len(lanes)} lane(s) are all expressible on {a.column}")
        return 0

    if a.degenerate:
        d = degenerate(a.column)
        for name in sorted(d):
            print(f"{name}\t{d[name]}")
        print(f"# {len(d)} of {len(s)} lanes are degenerate on {a.column} "
              f"({len(d) * len(_fixtures())} of {len(s) * len(_fixtures())} cells)")
        return 0

    if a.counts:
        nf = len(_fixtures())
        ib = identity_break()
        # TWO SCOPES, because the harness already trims one of them. A run
        # with no `--lanes` drops every `par-` lane (RECORD_EXCLUDED_PREFIXES,
        # enforced at tools/identity_break.py's lane selection), so the record
        # scope is what a full column costs TODAY; the whole registry is what
        # `--lanes` with a hand-written list can still ask for.
        for label, names in (("registry", sorted(s)), ("record-scope", sorted(ib.record_lanes()))):
            print(f"# {label}: {len(names)} lanes x {nf} fixtures")
            print("column\tlanes\tdegenerate\tshare\tcells\tdegenerate_cells")
            for c in sorted(COLUMNS):
                d = [n for n in names if not s[n].applicable(c)[0]]
                print(f"{c}\t{len(names)}\t{len(d)}\t{100.0 * len(d) / len(names):.1f}%\t"
                      f"{len(names) * nf}\t{len(d) * nf}")
            print()
        return 0

    if a.oracle_counts:
        ib = identity_break()
        rec = set(ib.record_lanes())
        live_batch = set(k for k, v in ib.BATCH.items() if not isinstance(v, str))
        print("# THE IN-CELL ORACLE (what raises inside one cell)")
        by = collections.Counter(sc.oracle for sc in s.values())
        for k in ("independent", "self", "recorded"):
            print(f"{k}\t{by[k]}\t{100.0 * by[k] / len(s):.1f}%")
        print(f"total\t{len(s)}")
        print("\n# WHAT ACTUALLY HOLDS THE LANE DOWN")
        print("anchor\tall\tin_record\tlive_batch_part")
        for k in ("in-cell-independent", "in-cell-self", "cross-route", "recorded-only"):
            names = [n for n in s if s[n].anchor == k]
            print(f"{k}\t{len(names)}\t{sum(1 for n in names if n in rec)}\t"
                  f"{sum(1 for n in names if n in live_batch)}")
        return 0

    rows = table()
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    cols = sorted(COLUMNS)
    print("lane\tkind\toracle\tanchor\tclaim_devices\tcpu_route\tin_record\t" + "\t".join(cols))
    for r in rows:
        print(f"{r['lane']}\t{r['kind']}\t{r['oracle']}\t{r['anchor']}\t{r['claim_devices']}\t"
              f"{'yes' if r['cpu_route'] else 'no'}\t{'yes' if r['record'] else 'no'}\t"
              + "\t".join(r["columns"][c] for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
