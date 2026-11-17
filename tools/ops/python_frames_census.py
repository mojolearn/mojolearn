#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Python frames census: the Python that actually RUNS inside each public call of an INSTALLED wheel.

AGENTS.md "No Python in the runtime": a public method is ONE binding call, and runtime behaviour, never a source
count, establishes compliance. This tool generalizes python/mojolearn/tests/test_public_call_no_mode_wrappers.py
(one estimator per family) to every public class member and public function: each is called under `sys.setprofile`
and the census records the Python functions entered inside the call (module:qualname, count; the public method's
own frame excluded, wrapper frames before it included) and how many native binding calls the call made.

Fixtures are verify's own, plus the census's own for the members verify never reaches (census_fixtures):
  api:<public>   one process per public class (or public function): the bounded endpoint fixtures of
                 `python -m mojolearn verify` (_verify_api_runner.registry(), the family checkers of
                 _verify_api_*.py, routed exactly as _verify_api_runner.capture() routes them). An estimator class
                 the estimator fixture applies to is also run through it for every one of its members in
                 _verify_api_estimators.METHODS (fit, transform, predict_proba, ...; predict through the
                 fixture's consumed-output read), and a class with get_params through the metadata fixture.
  light:<case>   one process per light tape case of verify (_verify_light.EXPECTED_CASES: ops.Session, the
                 training ops and optimizers on at most sixteen input elements).
  lane:<lane>    one process per CPU lane of verify's identity harness (mojolearn._identity_break, the lanes the
                 smoke's `lanes` step runs): the lane's fit on the base fixture, then the harness's own
                 _probe_fit (infer, model bytes, reload) and _probe_batch.
  fixture:<m>.<f> one process per census fixture (tools/ops/census_fixtures/<m>.py, fb4-census 2026-11-17): the
                 smallest working call of each public member verify's fixtures never reach (a tiny model, a two-worker
                 pool on the CPU wheel, a model file saved inside the fixture). Testing tools, never product: each
                 module imports only the standard library at load and takes `ml` (the installed mojolearn) and a
                 context `cx` (cx.tmp a fresh directory; cx.access and cx.delete measure a descriptor by access and a
                 finalizer by del). A member that cannot run on a CPU-only runner is listed in its module's NEEDS_GPU
                 with the reason (status NEEDS_GPU, counted apart from UNCOVERED, and listed in needs_gpu.tsv for the
                 GPU box's smoke; a fixture marked "gpu" runs only where MOJOLEARN_CENSUS_GPU=1); a member left
                 unreached on purpose is listed in WHY_NOT with the reason (uncovered.tsv).
After its fixtures, every unit PROBES the members they did not reach (py-census-14: 323 members UNCOVERED in
d95f805cce): on the last FITTED instance of each public class the fixtures used (the last instance a fit-named member
ran on; else the last instance: verify-fixtures-env, 2026-11-16, an unfitted prototype or a fresh metadata instance
used last had drawn 60 "not fitted" refusals), it reads each public property and calls
each public method that takes no argument (never a fit, train, step, load, save, export or run name; close, free
and discard last), then releases those instances under the profiler so their `__del__` is measured. Each probe is
listed as a fixture (`probe <member>`). A failed fixture is counted in one of three classes, each failure listed with
its class in failures.tsv: cpu_refusals (a GPU-only operation refusing on the CPU runner, _CPU_REFUSALS),
probe_refusals (a probe the product refused in the instance's state: unfitted, closed, a conditional member; the
member still ran) and fixture_errors (every other failure: a fixture defect or a product defect, read failures.tsv).
Packaging helpers (PACKAGING_FAMILIES: the install-metadata and surface
report modules no public operation calls) are PACKAGING, not UNCOVERED.
A public call is the outermost entry into product code (python/mojolearn outside tests/, verify and the harness)
made while no other public call is active: a Python frame of a product file, or a C call of a member installed as
a native function. A member no fixture reaches is listed UNCOVERED, never skipped.

EXEMPT BY RULE (AGENTS.md "scikit-learn tags"): `__sklearn_tags__` and the helpers used solely to build its tag
metadata (TAG_ROOTS and everything they call) are tagged `exempt` and not counted. Frames under the import system
(a lazy import inside a call) are tagged `import` and not counted either: they run once per process, not per call.
Product frames entered while an import runs outside any public call (a family's import-time connections, the
package's meta-path finders consulted for every import) are import time, never a public call: counted apart as
`import_time` (py-census-30, 2026-11-15: they had been read as calls and attributed to the first member they reached).

ATTRIBUTION. An outermost product frame that is not a public member (a decorator wrapper, or a private helper the
fixture calls) adopts the member it calls only when that member's frame is its direct child and the wrapper is not a
private (single underscore) name; any other public member reached inside it, and a public class's `__del__` reached
anywhere (a finalizer the collector runs inside whatever call is active), is measured as its own call while the outer
one waits. The unattributed entry keeps its own frames (unattributed.tsv, `inside`). A member never called itself but
whose defining function ran through a subclass's member is VIA_SUBCLASS (a mixin's or protocol's method), not
UNCOVERED.

SUMMARY: the first line leads with measured / with_python / uncovered / needs_gpu, so "0 with Python" is always read
beside what was not reached.

Subcommands (run by `tools/ops/wheel_verify_smoke.py frames`, which owns the parallel pool):
  inventory --surface FILE --out DIR   in the wheel's python: resolve the members and plan the units
  unit --inventory FILE --unit NAME --out DIR   in the wheel's python: run one unit, write DIR/<unit>.json
  report --dir DIR                     anywhere: census.tsv, functions.tsv, unattributed.tsv, SUMMARY lines
  run --python PY --out OUT            convenience: wheel_verify_smoke.py frames --python PY --out OUT

The inventory and unit subcommands import only the standard library and the installed `mojolearn`, so they run
under `python -I` from an empty directory. Python threads started inside a call are not profiled (setprofile is
per thread); the rule forbids Python workers, and a member that starts one shows the thread start as a frame.
"""
import argparse
import collections
import json
import os
import re
import sys
import time
import types
from pathlib import Path

#: `__sklearn_tags__` and its tag-only helpers (AGENTS.md "scikit-learn tags"): a frame of one of these names and
#: everything it calls is exempt. `_sk_validate._input_tag` and `_target_tag` are NOT here: input validation in
#: fit reads them, so they run as product work; called beneath a root below they are exempt with it.
TAG_ROOTS = frozenset(("__sklearn_tags__", "sklearn_tags", "_sklearn_instance_target_tags"))
#: public protocol dunders counted as members beside the public (non-underscore) names
MEMBER_DUNDERS = ("__init__", "__call__", "__getstate__", "__setstate__", "__del__")
#: package files that are fixtures, not product: tests, verification, the identity harness copy
_FIXTURE_PREFIXES = ("_verify", "_verification", "_identity")
_FIXTURE_DIRS = ("tests", "verify_reference")
TOP_FUNCTIONS = 30
#: families of package helpers that are installation and surface metadata, never called by a public operation:
#: `gpu_plugins` (which wheel and plugin to install, its markers and commands) and `host_surface` (the generated
#: inference-surface report). Listed as PACKAGING beside the census, never UNCOVERED (py-census-14).
PACKAGING_FAMILIES = frozenset(("gpu_plugins", "host_surface"))
#: zero-argument public methods a probe never calls: they fit, train, step, load, save, export or run work whose
#: inputs only a fixture can supply.
_PROBE_SKIP = ("fit", "partial_fit", "train", "step", "load", "save", "export", "install", "run", "main", "backward",
               "forward", "generate", "prefetch")
#: refusal texts of GPU-only operations on a CPU runner (multi-GPU drivers, device-resident optimizer entries):
#: counted as cpu_refusals beside fixture_errors (py-census-14: 51 of the 147 in d95f805cce). "has no device I/O
#: entries": a to_device member on a host binding (Embedding.to_device in 23bc7a5ae3).
_CPU_REFUSALS = ("no CPU implementation of the cooperative multi-GPU driver", "(a CPU-only install)",
                 "this binding has no device I/O entries")
#: member names that mark an instance FITTED for the probe (it probes the last fitted instance of a class)
_FIT_NAMES = ("fit", "partial_fit", "train", "search")
#: classes the estimator/metadata fixtures are never widened to (the census's own widening, never verify's routing):
#: a program that only a session builds (ops.Program) and a model only a trained GBDT text builds (HostGBDT). Their
#: members are reached through the classes and lanes that build them; widening them only drew fixture errors
#: (23bc7a5ae3: 4).
_NOT_WIDENED = ("mojolearn.ops.Program", "mojolearn.HostGBDT")
#: probed last: they end the instance's session.
_PROBE_LAST = ("close", "free", "discard")
#: the census's own fixtures (one module per family group; see the module docstring)
FIXTURE_DIR = Path(__file__).resolve().parent / "census_fixtures"


def fixture_modules(directory=FIXTURE_DIR):
    """{module name: module} of the census fixture modules, loaded by path (the unit runs under `python -I`, where
    the tool's own directory is not on sys.path). Loading imports only the standard library."""
    import importlib.util
    mods = {}
    for path in sorted(Path(directory).glob("[a-z]*.py")):
        spec = importlib.util.spec_from_file_location(f"_census_fixtures_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[path.stem] = mod
    return mods


def fixture_plan(directory=FIXTURE_DIR):
    """(fixtures, needs_gpu, why_not): fixtures {unit id "<module>.<name>": (fn, members, gpu)}, needs_gpu and
    why_not {member: reason}. A module's FIXTURES maps a name to (fn, [members]) or (fn, [members], "gpu")."""
    fixtures, needs_gpu, why_not = {}, {}, {}
    for mname, mod in fixture_modules(directory).items():
        for name, entry in getattr(mod, "FIXTURES", {}).items():
            fn, members = entry[0], list(entry[1])
            fixtures[f"{mname}.{name}"] = (fn, members, len(entry) > 2 and entry[2] == "gpu")
        needs_gpu.update(getattr(mod, "NEEDS_GPU", {}))
        why_not.update(getattr(mod, "WHY_NOT", {}))
    return fixtures, needs_gpu, why_not


# --------------------------------------------------------------------------------------------------------------
# inventory (wheel python)
# --------------------------------------------------------------------------------------------------------------

def _member_kind(cls, attr):
    """How `cls.attr` runs, or None when it is not a callable member defined by the package (a plain value, an
    instance attribute, or an attribute inherited from outside mojolearn)."""
    owner = next((k for k in cls.__mro__ if attr in vars(k)), None)
    if owner is None or not (getattr(owner, "__module__", "") or "").startswith("mojolearn"):
        return None
    raw = vars(owner)[attr]
    if isinstance(raw, (staticmethod, classmethod)):
        raw = raw.__func__
    if isinstance(raw, property):
        return "property"
    if isinstance(raw, types.FunctionType):
        return "method"
    if isinstance(raw, (types.GetSetDescriptorType, types.MemberDescriptorType)):
        return "native_attribute"
    if isinstance(raw, (types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType,
                        types.ClassMethodDescriptorType)):
        return "native"
    if hasattr(type(raw), "__get__") and callable(getattr(type(raw), "__get__")) and not isinstance(raw, type):
        return "descriptor"
    if callable(raw) and not isinstance(raw, type):
        return "callable"
    return None


def _function_key(raw):
    """"<module>.<qualname>" of the Python function a class attribute runs, or "" (a native member): the same key
    for every class that installs one function (randomforest's `save = ForestArchive.save`), so a member whose
    function ran through another class's member reads VIA_SUBCLASS."""
    if isinstance(raw, (staticmethod, classmethod)):
        raw = raw.__func__
    if isinstance(raw, property):
        raw = raw.fget
    if isinstance(raw, types.FunctionType):
        return f"{raw.__module__}.{raw.__qualname__}"
    return ""


def _family(paths, obj):
    """The public submodule a member is exported from (linear_model, ensemble, nn, ...); for a name exported only
    at the top level, its defining module without the package prefix, leading underscores and the internal
    expansion_/native_/x_ prefixes and _impl suffix."""
    subs = sorted(p.split(".")[1] for p in paths if p.count(".") >= 2)
    if subs:
        return subs[0]
    mod = (getattr(obj, "__module__", "") or "").split(".")
    name = mod[1] if len(mod) > 1 and mod[0] == "mojolearn" else (mod[0] if mod else "")
    name = name.lstrip("_")
    for prefix in ("expansion_", "native_", "x_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return re.sub(r"_impl$", "", name) or "mojolearn"


def cmd_inventory(args):
    import importlib
    import inspect
    import mojolearn
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exports, static_methods = [], collections.defaultdict(set)
    for line in Path(args.surface).read_text().splitlines():
        kind, _, rest = line.partition(":")
        if kind == "export":
            mod, _, name = rest.rpartition(".")
            exports.append((mod, name))
        elif kind == "method":
            cname, _, attr = rest.partition(".")
            static_methods[cname].add(attr)
    # The static surface reads literal `__all__` lists only; a package that extends `__all__` at import (the lazy
    # expansion doors' names in mojolearn/__init__.py) exports more than it lists. Add the live `__all__` of every
    # module the surface names (py-census-30: BayesianRidge and the other door classes were missing).
    listed = set(exports)
    for mod in sorted({m for m, _ in exports} | {"mojolearn"}):
        modname = "mojolearn" if mod == "mojolearn" else "mojolearn." + mod
        try:
            live = tuple(getattr(importlib.import_module(modname), "__all__", ()) or ())
        except Exception:   # the export loop below records the module's error once per listed name
            continue
        for name in live:
            if isinstance(name, str) and (mod, name) not in listed:
                listed.add((mod, name))
                exports.append((mod, name))
    errors, objects = [], {}
    for mod, name in exports:
        modname = "mojolearn" if mod == "mojolearn" else "mojolearn." + mod
        try:
            obj = getattr(importlib.import_module(modname), name)
        except Exception as exc:   # an export this install cannot resolve: listed, never a census failure
            errors.append(f"{modname}.{name}\t{type(exc).__name__}: {str(exc)[:200]}")
            continue
        path = modname + "." + name
        entry = objects.setdefault(id(obj), [obj, []])
        entry[1].append(path)
    classes, functions, members = [], [], []
    for obj, paths in objects.values():
        canonical = min(paths, key=lambda p: (p.count("."), p))
        family = _family(paths, obj)
        module = getattr(obj, "__module__", "") or ""
        if isinstance(obj, type):
            if not module.startswith("mojolearn"):
                continue
            try:
                listed = set(dir(obj))
            except Exception:
                listed = set()
            names = set(static_methods.get(obj.__name__, ())) | {a for a in listed if not a.startswith("_")}
            names |= {d for d in MEMBER_DUNDERS if d in listed}
            attrs, defined, codes = {}, {}, {}
            for attr in sorted(names):
                if attr.startswith("_") and attr not in MEMBER_DUNDERS:
                    continue
                try:
                    kind = _member_kind(obj, attr)
                except Exception:
                    kind = None
                if kind is not None:
                    attrs[attr] = kind
                    owner = next(k for k in obj.__mro__ if attr in vars(k))
                    defined[attr] = f"{owner.__module__}.{owner.__qualname__}.{attr}"
                    codes[attr] = _function_key(vars(owner)[attr])
            classes.append(dict(id=canonical, module=module, qualname=obj.__qualname__, family=family,
                                aliases=sorted(paths), attrs=attrs))
            members += [dict(id=f"{canonical}.{a}", family=family, kind=k, defined=defined[a],
                             **({"code": codes[a]} if codes.get(a) else {}))
                        for a, k in attrs.items()]
        elif callable(obj) and (module.startswith("mojolearn") or isinstance(obj, types.BuiltinFunctionType)):
            if isinstance(obj, types.FunctionType):
                kind, qual = "function", obj.__qualname__
            elif isinstance(obj, types.BuiltinFunctionType):
                owner = getattr(obj, "__self__", None)
                kind = "native_function"
                module = getattr(owner, "__name__", module) if isinstance(owner, types.ModuleType) else module
                qual = obj.__name__
                if not (module or "").startswith(("mojolearn", "_mojolearn")):
                    continue
            else:
                continue
            functions.append(dict(id=canonical, module=module, qualname=qual, family=family, kind=kind,
                                  aliases=sorted(paths)))
            members.append(dict(id=canonical, family=family, kind=kind, defined=canonical))
    # the units: verify's endpoint fixtures per public object, the harness lanes per lane
    units = []
    from mojolearn import _verify_api_runner as runner, _verify_api_estimators as est
    from mojolearn._verification_api_targets import TARGETS
    routed = {r["public"] for r in TARGETS}
    by_path = {}
    for c in classes:
        for p in c["aliases"]:
            by_path[p] = c
    for f in functions:
        for p in f["aliases"]:
            by_path[p] = f
    groups = collections.defaultdict(set)
    for public in routed:
        owner = by_path.get(public)
        groups[owner["id"] if owner else public].add(public)
    for c in classes:
        attrs = set(c["attrs"])
        routed_class = any(p in routed for p in c["aliases"])
        # A protocol base (ForestProtocol) is never constructed itself: its members run through the estimators that
        # inherit them (VIA_SUBCLASS). Driving it as an estimator only failed (py-census-14: 9 fixture errors).
        if c["qualname"].endswith("Protocol") and not routed_class:
            continue
        if (est.applicable(c["id"], sorted(attrs & est.METHODS)) or "get_params" in attrs or routed_class):
            groups[c["id"]].add(c["id"])
    for uid, publics in sorted(groups.items()):
        units.append(dict(name="api:" + uid, kind="api", id=uid, publics=sorted(publics)))
    try:   # verify's light tape cases (ops.Session and the training ops), one unit per case
        light = importlib.import_module("mojolearn._verify_light")
        units += [dict(name="light:" + c, kind="light", id=c) for c in light.EXPECTED_CASES]
    except Exception as exc:
        errors.append(f"mojolearn._verify_light\t{type(exc).__name__}: {str(exc)[:200]}")
    try:
        h = importlib.import_module("mojolearn._identity_break")
        lanes = [n for n in h.LANES if not n.startswith(h.RECORD_EXCLUDED_PREFIXES) and n not in h.GPU_ONLY_LANES]
    except Exception as exc:
        errors.append(f"mojolearn._identity_break\t{type(exc).__name__}: {str(exc)[:200]}")
        lanes = []
    for lane in lanes:
        try:
            src = inspect.getsource(h.LANES[lane])
        except (OSError, TypeError):
            src = ""
        refs = sorted(set(re.findall(r"\bml\.([A-Z]\w*)", src)))
        units.append(dict(name="lane:" + lane, kind="lane", id=lane, classes=refs))
    # the census's own fixtures (tools/ops/census_fixtures): one unit per fixture, a "gpu" fixture only where
    # MOJOLEARN_CENSUS_GPU=1 (the GPU box's smoke); NEEDS_GPU and WHY_NOT ride in the inventory for the report
    needs_gpu, why_not = {}, {}
    try:
        fixtures, needs_gpu, why_not = fixture_plan()
        gpu = os.environ.get("MOJOLEARN_CENSUS_GPU") == "1"
        for fid, (_, fmembers, needs) in sorted(fixtures.items()):
            if needs and not gpu:
                continue
            units.append(dict(name="fixture:" + fid, kind="fixture", id=fid, members=fmembers))
    except Exception as exc:
        errors.append(f"census_fixtures\t{type(exc).__name__}: {str(exc)[:200]}")
    inv = dict(format="mojolearn.python-frames-inventory.v1", version=mojolearn.__version__,
               package=os.path.dirname(os.path.abspath(mojolearn.__file__)), classes=classes,
               functions=functions, members=members, units=units, export_errors=errors, needs_gpu=needs_gpu,
               why_not=why_not)
    (out / "inventory.json").write_text(json.dumps(inv, indent=1, sort_keys=True) + "\n")
    print(f"inventory: {len(members)} members ({len(classes)} classes, {len(functions)} functions), "
          f"{len(units)} units ({len(lanes)} lanes, {sum(u['kind'] == 'fixture' for u in units)} census fixtures), "
          f"{len(errors)} export errors", flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------
# the profiler (wheel python)
# --------------------------------------------------------------------------------------------------------------

class _Call:
    __slots__ = ("kind", "entry", "carg_name", "carg_self", "mids", "pre", "funcs", "n", "bindings", "bnames",
                 "exempt", "imports", "exempt_root", "import_root", "fallback")

    def __init__(self, kind, entry):
        self.kind, self.entry = kind, entry
        self.carg_name = self.carg_self = None
        self.mids = None
        self.pre = []
        self.funcs = collections.Counter()
        self.n = self.bindings = self.exempt = self.imports = 0
        self.bnames = collections.Counter()
        self.exempt_root = self.import_root = None
        self.fallback = None   # (member ids, native reader) of a probe's own read (Census.open_member)


def _new_stats():
    return dict(calls=0, zero=0, frames=0, frames_max=0, bindings_max=0, bindings_total=0, multi_binding=0,
                no_binding=0, exempt=0, imports=0, funcs=collections.Counter(), bnames=collections.Counter())


class Census:
    """Per-process profiler state: public calls seen and the Python each one ran."""

    def __init__(self, inv):
        import importlib.machinery
        self.pkg = inv["package"]
        self.pkg_real = os.path.realpath(self.pkg)
        self.site = os.path.dirname(self.pkg_real)
        self.ext = tuple(importlib.machinery.EXTENSION_SUFFIXES) + (".so", ".pyd", ".dylib")
        self.classes = {(c["module"], c["qualname"]): c for c in inv["classes"]}
        self.class_ids = {c["id"]: c for c in inv["classes"]}
        self.instances = {}            # class id -> the last instance a member call ran on (probe_members)
        self.fitted = {}               # class id -> the last instance a fit-named member ran on (probed first)
        self.functions = {(f["module"], f["qualname"]): f["id"] for f in inv["functions"]}
        # every member by the function it defines ("<module>.<qualname>"): a static or class method whose first
        # argument is no public instance (ConstantLR.from_config(dict)), or a member's function reached through a
        # non-public owner, is credited by its own function (fb4-census: from_config stayed UNCOVERED)
        self.by_defined = collections.defaultdict(list)
        for m in inv["members"]:
            if m.get("kind") in ("method", "property") and m.get("defined"):
                self.by_defined[m["defined"]].append(m["id"])
        self.members = {}
        self.unattributed = collections.Counter()
        self.where_cache, self.key_cache = {}, {}
        self.type_cache, self.code_maps, self.native_names = {}, {}, {}
        self.native_ids = collections.defaultdict(set)
        self.binding_cache = {}
        self.type_modules = {}         # id(type) -> (type, module name) for native types without __module__
        self.seen = set()
        self.call = None
        self.suspended = []            # outer calls waiting while a member reached inside them runs as its own
        self.import_frame = None       # the outermost import frame entered while no call is active
        self.import_time = collections.Counter()
        self.unattributed_inside = collections.Counter()
        self.errors = collections.Counter()

    # ---- classification -------------------------------------------------------------------------------------
    def where(self, code):
        """1 product, 2 fixture (tests, verification, harness), 0 outside the package."""
        w = self.where_cache.get(code)
        if w is None:
            fn = code.co_filename
            w = 0
            for root in (self.pkg, self.pkg_real):
                if fn.startswith(root + os.sep):
                    rel = fn[len(root) + 1:].split(os.sep)
                    w = 2 if (rel[0] in _FIXTURE_DIRS or rel[-1].startswith(_FIXTURE_PREFIXES)
                              or rel[0].startswith(_FIXTURE_PREFIXES)) else 1
                    break
            self.where_cache[code] = w
        return w

    def key(self, code, frame):
        k = self.key_cache.get(code)
        if k is None:
            k = f"{frame.f_globals.get('__name__', '?')}:{getattr(code, 'co_qualname', code.co_name)}"
            self.key_cache[code] = k
        return k

    def public_class(self, T):
        """(class record, the public class object in T's MRO) or None."""
        if T in self.type_cache:
            return self.type_cache[T]
        found = None
        for K in getattr(T, "__mro__", ()):
            c = self.classes.get((getattr(K, "__module__", None), getattr(K, "__qualname__", None)))
            if c is not None:
                found = (c, K)
                break
        self.type_cache[T] = found
        return found

    def codes_for(self, record, K):
        """{code: [member ids]} of a public class (innermost function of each member, unwrapped from its
        decorators), built once per class; registers its native members too."""
        cid = record["id"]
        m = self.code_maps.get(cid)
        if m is not None:
            return m
        import inspect
        m = collections.defaultdict(list)
        names = collections.defaultdict(list)
        for attr, kind in record["attrs"].items():
            mid = f"{cid}.{attr}"
            try:
                raw = inspect.getattr_static(K, attr)
            except AttributeError:
                continue
            if kind in ("native", "callable"):
                try:
                    value = getattr(K, attr)
                except Exception:
                    value = raw
                self.native_ids[id(value)].add(mid)
                names[getattr(value, "__name__", attr)].append(mid)
                if getattr(value, "__name__", attr) != attr:
                    names[attr].append(mid)
                continue
            for code in _codes(raw):
                if mid not in m[code]:
                    m[code].append(mid)
        self.code_maps[cid] = m
        self.native_names[cid] = names
        self.seen.add(cid)
        return m

    def prebuild(self, cls):
        pc = self.public_class(cls)
        if pc:
            self.codes_for(*pc)

    def resolve(self, frame, code):
        """The member ids whose own function this frame runs, or None."""
        if code.co_argcount or code.co_posonlyargcount:
            try:
                obj = frame.f_locals.get(code.co_varnames[0])
            except Exception:
                obj = None
            if obj is not None:
                pc = self.public_class(obj if isinstance(obj, type) else type(obj))
                if pc is not None:
                    mids = self.codes_for(*pc).get(code)
                    if mids:
                        if not isinstance(obj, type):
                            self.note_instance(pc[0]["id"], obj, mids)
                        return mids
        name, qual = frame.f_globals.get("__name__"), getattr(code, "co_qualname", code.co_name)
        fid = self.functions.get((name, qual))
        if fid:
            return [fid]
        return list(self.by_defined.get(f"{name}.{qual}", ())) or None

    def note_instance(self, cid, obj, mids):
        """The instance a member call ran on, for the probe; a fit-named member marks it fitted too."""
        self.instances[cid] = obj
        if any(m.rsplit(".", 1)[-1].startswith(_FIT_NAMES) for m in mids):
            self.fitted[cid] = obj

    def native_member(self, arg):
        owner = getattr(arg, "__self__", None)
        name = getattr(arg, "__name__", None)
        if owner is not None and not isinstance(owner, types.ModuleType):
            pc = self.public_class(owner if isinstance(owner, type) else type(owner))
            if pc is not None:
                self.codes_for(*pc)
                mids = self.native_names.get(pc[0]["id"], {}).get(name)
                if mids:
                    if not isinstance(owner, type):
                        self.note_instance(pc[0]["id"], owner, mids)
                    return list(mids)
        cands = self.native_ids.get(id(arg))
        if cands:
            seen = [m for m in cands if m.rsplit(".", 1)[0] in self.seen]
            if len(cands) == 1:
                return list(cands)
            if len(seen) == 1:
                return seen
        if isinstance(owner, types.ModuleType):
            fid = self.functions.get((owner.__name__, name))
            if fid:
                return [fid]
        if owner is not None:
            # A native bound method a public instance holds in its own dictionary, shadowing the class member of
            # that name (a schedule's `lr_at` / `bits_at` are its native rate block's methods, bound into the
            # schedule at construction; fb6-smoke, 2026-11-17: they read UNCOVERED). The callable itself is the
            # stored object, so identity with an instance entry attributes the call to the member of that entry's name.
            for cid, inst in list(self.instances.items()):
                try:
                    fields = object.__getattribute__(inst, "__dict__")
                except Exception:
                    continue
                if type(fields) is not dict:
                    continue
                attrs = self.class_ids[cid]["attrs"]
                held = next((k for k, v in fields.items() if v is arg and k in attrs), None)
                if held is not None:
                    return [f"{cid}.{held}"]
        return None

    def type_module(self, T):
        """T's module name; for a native type registered under an undotted name (no __module__: the schedule rate
        block `_LrScheduleBlock`), the loaded module that holds the type object, once per type."""
        name = getattr(T, "__module__", None)
        if name:
            return name
        cache = self.type_modules
        if id(T) not in cache:
            holders = [m.__name__ for m in list(sys.modules.values())
                       if isinstance(m, types.ModuleType) and any(v is T for v in list(vars(m).values()))]
            ours = [h for h in holders if h.lstrip("_").startswith("mojolearn")]
            cache[id(T)] = (T, (ours or holders or [None])[0])
        return cache[id(T)][1]

    def binding(self, arg):
        """(is a native mojolearn binding, its name) for a C callable."""
        owner = getattr(arg, "__self__", None)
        if isinstance(owner, types.ModuleType):
            mod, modname = owner, owner.__name__
        else:
            modname = getattr(arg, "__module__", None) or (self.type_module(type(owner)) if owner is not None else None)
            mod = sys.modules.get(modname) if modname else None
        ck = id(mod) if mod is not None else modname
        hit = self.binding_cache.get(ck)
        if hit is None:
            raw = getattr(mod, "__file__", None) if mod is not None else None
            fn = os.path.realpath(raw) if isinstance(raw, str) and raw else ""
            native_file = fn.endswith(self.ext)
            ours = (fn.startswith(self.site + os.sep)
                    and fn[len(self.site) + 1:].split(os.sep)[0].lstrip("_").startswith("mojolearn"))
            # a C callable of a module (or native type) named mojolearn*/_mojolearn*, or of an extension file
            # installed under a mojolearn* directory of site-packages
            named = (modname or "").lstrip("_").startswith("mojolearn")
            hit = named or (native_file and ours)
            self.binding_cache[ck] = hit
        return hit, f"{modname}.{getattr(arg, '__name__', '?')}"

    # ---- the profile function ---------------------------------------------------------------------------------
    def profile(self, frame, event, arg):
        try:
            call = self.call
            if event == "call":
                code = frame.f_code
                if call is None:
                    if self.import_frame is not None:   # import time: never a public call
                        if self.where(code) == 1:
                            self.import_time[self.key(code, frame)] += 1
                        return
                    if _is_import(code.co_filename):
                        self.import_frame = frame
                        return
                    # a module body runs once, at import: never a public call
                    if code.co_name != "<module>" and self.where(code) == 1:
                        self.start(frame, code)
                    return
                self.inner(call, frame, code)
            elif event == "return":
                if call is None:
                    if frame is self.import_frame:
                        self.import_frame = None
                    return
                if call.exempt_root is frame:
                    call.exempt_root = None
                elif call.import_root is frame:
                    call.import_root = None
                if call.kind == "py" and frame is call.entry:
                    self.finish(call)
            elif event == "c_call":
                if call is None:
                    if self.import_frame is None and self.where(frame.f_code) != 1:
                        mids = self.native_member(arg)
                        if mids:
                            call = self.call = _Call("c", frame)
                            call.carg_name = getattr(arg, "__name__", None)
                            call.carg_self = getattr(arg, "__self__", None)
                            call.mids = mids
                            hit, name = self.binding(arg)
                            if hit:
                                call.bindings, call.bnames[name] = 1, 1
                    return
                if call.exempt_root is None and call.import_root is None:
                    hit, name = self.binding(arg)
                    if hit:
                        call.bindings += 1
                        call.bnames[name] += 1
            elif event in ("c_return", "c_exception"):
                if (call is not None and call.kind == "c" and frame is call.entry
                        and getattr(arg, "__name__", None) == call.carg_name
                        and getattr(arg, "__self__", None) is call.carg_self):
                    self.finish(call)
        except Exception as exc:   # the profiler must never raise into the call it measures
            self.errors[type(exc).__name__] += 1

    def start(self, frame, code):
        call = self.call = _Call("py", frame)
        if code.co_name in TAG_ROOTS:   # the fixture asked for the tags themselves: exempt, not a public call
            call.exempt_root = frame
            return
        call.mids = self.resolve(frame, code)
        if call.mids is None:
            call.pre.append(self.key(code, frame))

    def inner(self, call, frame, code):
        if call.exempt_root is not None:
            call.exempt += 1
            return
        if call.import_root is not None:
            call.imports += 1
            return
        if code.co_name in TAG_ROOTS:
            call.exempt_root = frame
            call.exempt += 1
            return
        if _is_import(code.co_filename):
            call.import_root = frame
            call.imports += 1
            return
        finalizer = code.co_name == "__del__"
        if (call.mids is None or finalizer) and self.where(code) == 1:
            mids = self.resolve(frame, code)
            if mids:
                entry = call.entry.f_code.co_name if call.kind == "py" else ""
                if (call.mids is None and not finalizer and frame.f_back is call.entry
                        and not _private(entry)):
                    call.mids = mids   # the public method's own frame under its wrapper: not counted
                    return
                # a member reached inside a private helper or deeper inside a wrapper, or a finalizer: its own call
                self.suspended.append(call)
                nested = self.call = _Call("py", frame)
                nested.mids = mids
                return
        call.funcs[self.key(code, frame)] += 1
        call.n += 1

    def open_member(self, frame, mids, reader):
        """Make a probe's own frame the entry of a call to `mids`. A Python accessor below it adopts the member as
        any wrapper's child does (its own frame not counted). A property a class's NATIVE `__getattribute__`
        serves (the native public attribute plans, e.g. ARIMA.ar_ .. sigma2_, Cholesky.logdet_, SVC.coef_) runs
        no Python frame and no C call the profiler sees: until 2026-11-16 such a read was never a call, and the
        members stayed UNCOVERED although every probe of them succeeded (lane cpu-surface-defects). `finish`
        now credits the read to `mids`, with that native reader as its one binding call."""
        call = self.call = _Call("py", frame)
        call.fallback = (mids, reader)

    def finish(self, call):
        self.call = self.suspended.pop() if self.suspended else None
        if call.mids is None and call.fallback is not None:
            call.mids, reader = call.fallback
            if reader is not None and call.n == 0 and call.bindings == 0:
                # the class's native __getattribute__ (a compiled descriptor the package installed) is the one
                # native call the read made; its type names it
                call.bindings = 1
                call.bnames[f"{type(reader).__module__}.{type(reader).__qualname__} (native __getattribute__)"] += 1
        if call.mids is None:
            for k in call.pre:
                self.unattributed[k] += 1
            self.unattributed_inside.update(call.funcs)
            return
        for k in call.pre:   # a wrapper entered before the public method's own frame
            call.funcs[k] += 1
            call.n += 1
        for mid in call.mids:
            st = self.members.get(mid)
            if st is None:
                st = self.members[mid] = _new_stats()
            st["calls"] += 1
            st["zero"] += call.n == 0
            st["frames"] += call.n
            st["frames_max"] = max(st["frames_max"], call.n)
            st["bindings_max"] = max(st["bindings_max"], call.bindings)
            st["bindings_total"] += call.bindings
            st["multi_binding"] += call.bindings > 1
            st["no_binding"] += call.bindings == 0
            st["exempt"] += call.exempt
            st["imports"] += call.imports
            st["funcs"].update(call.funcs)
            st["bnames"].update(call.bnames)

    def run(self, operation):
        """operation() under the profiler; (ok, error text)."""
        previous = sys.getprofile()
        sys.setprofile(self.profile)
        try:
            operation()
            return True, None
        except BaseException as exc:   # a fixture that fails still leaves the calls it made recorded
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return False, f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            sys.setprofile(previous)
            while self.call is not None:   # unwound without its return event (should not happen): close it
                self.errors["unclosed_call"] += 1
                self.finish(self.call)
            self.import_frame = None


def _is_import(filename):
    return filename.startswith("<frozen importlib") or (os.sep + "importlib" + os.sep) in filename


def _private(name):
    """A single-underscore name (a private helper), not a dunder."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _codes(raw):
    """Code objects of the innermost functions a class attribute runs (property accessors included)."""
    if isinstance(raw, (staticmethod, classmethod)):
        raw = raw.__func__
    cands = [raw.fget, raw.fset, raw.fdel] if isinstance(raw, property) else [raw]
    # Every function along the wrapper chain: the installed wrapper runs first (a functools.wraps constructor such
    # as _forest_protocol's `initialize` was unmapped while only its __wrapped__ original was, so the forest
    # constructors read as unattributed entries and their __init__ as UNCOVERED); the original it wraps, when it
    # runs, is then a frame inside the call, as a wrapper frame before it used to be.
    out = []

    def add(fn):
        code = getattr(fn, "__code__", None)
        if code is not None and code not in out:
            out.append(code)
    for f in cands:
        for _ in range(8):   # descriptor wrappers (available_if and similar) hold the function as fn/func
            add(f)
            inner = next((getattr(f, a) for a in ("fn", "func", "__func__")
                          if isinstance(getattr(f, a, None), types.FunctionType)), None)
            if inner is None or inner is f:
                break
            f = inner
        seen = 0
        while getattr(f, "__wrapped__", None) is not None and seen < 16:
            add(f)
            f = f.__wrapped__
            seen += 1
        add(f)
    return out


# --------------------------------------------------------------------------------------------------------------
# one unit (wheel python)
# --------------------------------------------------------------------------------------------------------------

class _Recorder:
    """Stands in for _verify_api.Recorder: every endpoint the fixture asks for runs once, under the profiler."""

    def __init__(self, census, record, done, public=None, cls=None):
        self.census, self.record, self.done = census, record, done
        self.rows = {}
        self.public, self.cls = public, cls

    def check(self, endpoint, operation, *, kind="numerical"):
        if endpoint in self.rows:
            return self.rows[endpoint]
        # A fixture endpoint naming a member the class does not have is no member to measure: verify's Recorder
        # runs only its selected endpoints, this one runs every endpoint a fixture names (the metadata fixture's
        # set_params on the model_selection splitters, which have get_params only: 15 errors in 23bc7a5ae3).
        if self.cls is not None and endpoint.startswith(self.public + "."):
            member = endpoint[len(self.public) + 1:].split(":", 1)[0]
            if member and not hasattr(self.cls, member):
                return dict(status="absent", kind=kind)
        ok, error = self.census.run(operation)
        self.rows[endpoint] = dict(status="captured" if ok else "failed", kind=kind)
        self.record(endpoint, ok, error)
        self.done()
        return self.rows[endpoint]


def _probe_members(census, record, done):
    """The members this unit's fixtures did not reach, on the last instance of each public class they used: each
    public property read, each public method that takes no argument called (_PROBE_SKIP never; _PROBE_LAST last);
    then the instances released under the profiler, so their finalizers are measured."""
    import inspect
    # the last fitted instance of each class, else its last instance (an unfitted prototype or a fresh metadata
    # instance used after the fit would only draw "not fitted" refusals); both dicts released together below
    instances = dict(census.instances, **census.fitted)
    census.instances, census.fitted = {}, {}
    for cid in sorted(instances):
        cls = census.class_ids.get(cid)
        if cls is None:
            continue
        target = instances[cid]
        names = sorted((a for a, k in cls["attrs"].items() if not a.startswith("_") and k in ("property", "method")
                        and f"{cid}.{a}" not in census.members and not a.startswith(_PROBE_SKIP)),
                       key=lambda a: (a in _PROBE_LAST, a))
        for attr in names:
            if cls["attrs"][attr] == "property":
                # The class's own __getattribute__ when it is not a Python function: the native reader that
                # serves the property without a Python frame (Census.open_member).
                try:
                    reader = inspect.getattr_static(type(target), "__getattribute__")
                except AttributeError:
                    reader = None
                if isinstance(reader, types.FunctionType) or reader is object.__getattribute__:
                    reader = None

                if reader is None:   # a Python accessor runs: attributed by its own frame, as before
                    def operation(target=target, attr=attr):
                        getattr(target, attr)
                else:
                    def operation(target=target, attr=attr, mids=[f"{cid}.{attr}"], reader=reader):
                        census.open_member(sys._getframe(), mids, reader)
                        getattr(target, attr)
            else:
                try:
                    operation = getattr(target, attr)
                    required = [p for p in inspect.signature(operation).parameters.values()
                                if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
                except Exception:   # not a plain callable on this instance: nothing to probe
                    continue
                if required:
                    continue
            ok, error = census.run(operation)
            operation = None   # holds the instance: dropped so the release below runs its finalizer
            record(f"probe {cid}.{attr}", ok, error)
        target = None
    done()
    ok, error = census.run(instances.clear)
    record("release instances", ok, error)
    done()


def _api_unit(census, unit, record, done, inv_classes):
    from mojolearn import _verify_api_runner as runner, _verify_api_estimators as est
    from mojolearn import _verify_api_metadata as metadata
    from mojolearn._verification_api_targets import TARGETS
    from mojolearn._verify_api import resolve
    publics = set(unit["publics"])
    classes = {c["id"]: c for c in inv_classes}
    try:
        obj = resolve(unit["id"])
        if isinstance(obj, type):
            census.prebuild(obj)
    except Exception as exc:
        record(f"resolve {unit['id']}", False, f"{type(exc).__name__}: {str(exc)[:300]}")
        obj = None
    rec = _Recorder(census, record, done, unit["id"], obj if isinstance(obj, type) else None)
    routes = runner.registry()
    groups = {}
    for row in TARGETS:
        if row["public"] in publics and row["endpoint"] in routes:
            family, module, public, method = routes[row["endpoint"]]
            groups.setdefault((family, public), (module, []))[1].append(method)
    # verify's estimator and metadata fixtures, widened to every member they can drive on this class
    # (the class's own members from the inventory, so a default inherited from `object` is never driven)
    attrs = set(classes.get(unit["id"], {}).get("attrs", ()))
    extra = sorted(attrs & est.METHODS)
    if unit["id"] in _NOT_WIDENED or unit["id"].endswith("Mixin"):
        extra = []
    if extra and est.applicable(unit["id"], extra):
        key = next((k for k in groups if k[0] == "estimators"), ("estimators", unit["id"]))
        groups[key] = (est, sorted(set(groups.get(key, (est, []))[1]) | set(extra)))
    if "get_params" in attrs and not any(f == "metadata" for f, _ in groups):
        groups[("metadata", unit["id"])] = (metadata, [])
    names = {"estimators": "estimator", "optimizers": "optimizer", "neural": "neural", "training": "training",
             "programs": "program", "models": "model", "misc": "misc"}
    for (family, public), (module, methods) in sorted(groups.items(), key=lambda kv: kv[0]):
        def checks(family=family, public=public, module=module, methods=methods):
            if family == "metadata":
                module.parameter_checks(rec, public)
            elif family == "utilities":
                module.utility_checks(rec, [r for r in TARGETS if r["public"] == public
                                            and r["endpoint"] in routes and routes[r["endpoint"]][0] == family])
            else:
                getattr(module, names[family] + "_checks")(rec, public, methods)
        # The whole checker runs under the profiler (py-census-14): the constructors the fixture calls before an
        # endpoint and the finalizers of the objects it drops on return are measured too. A checker that fails
        # outside an endpoint is listed and the unit goes on.
        ok, error = census.run(checks)
        record(f"{family}:{public}", ok, error)
        done()
    _probe_members(census, record, done)


def _lane_unit(census, unit, record, done):
    import importlib
    import mojolearn as ml
    h = importlib.import_module("mojolearn._identity_break")
    lane = unit["id"]
    for name in unit.get("classes", ()):
        try:
            census.prebuild(getattr(ml, name))
        except Exception:
            pass
    X, yc, yr = h.fixture("base")
    held = h.heldout("base")
    h._DUMP_TAG = f"{lane}/base/0"
    box = {}

    def fit():
        box["fit"] = h.LANES[lane](ml, X, yc, yr, held.copy())
    ok, error = census.run(fit)
    record("fit", ok, error)
    done()
    if not ok:
        return
    for stage, op in (("probe_fit", lambda: h._probe_fit(box["fit"], lane)),
                      ("probe_batch", lambda: h._probe_batch(box["fit"], lane, ml, held.copy(), h.BATCH_ALONE,
                                                             False))):
        ok, error = census.run(op)
        record(stage, ok, error)
        done()
    ok, error = census.run(box.clear)   # the lane's fitted objects released under the profiler (finalizers)
    record("release fit", ok, error)
    _probe_members(census, record, done)


class FixtureContext:
    """What a census fixture gets beside `ml`: a fresh directory, and the two measurements a plain call cannot make.

    access(obj, attr, member, expect=())  reads obj.attr as a call of `member` (a descriptor: a native or
        `available_if` member whose read is the operation, or that refuses with AttributeError when the instance
        does not offer it; pass that exception in `expect`). The read is credited as Census.open_member credits a
        probe's read: Python frames under it are counted, a read served natively counts its reader as one binding.
    delete(holder, key, member)  drops holder[key] (the fixture's last reference) as a call of `member`, so the
        object's `__del__` runs under the profiler: a Python finalizer is counted by its frames, a native one (a
        compiled descriptor installed as __del__, which CPython calls from C, unseen by the profiler) is credited
        with that descriptor as its one binding. The census's own references to the object (the probe's instance
        records) are dropped first so the del is the last one."""

    def __init__(self, census, tmp):
        self.census, self.tmp = census, Path(tmp)

    def access(self, obj, attr, member, expect=()):
        import inspect
        try:
            reader = inspect.getattr_static(type(obj), "__getattribute__")
        except AttributeError:
            reader = None
        if isinstance(reader, types.FunctionType) or reader is object.__getattribute__:
            reader = None
        return self._read(obj, attr, [member], reader, tuple(expect))

    def _read(self, obj, attr, mids, reader, expect):
        self.census.open_member(sys._getframe(), mids, reader)
        try:
            return getattr(obj, attr)
        except expect:
            return None

    def delete(self, holder, key, member):
        import gc
        import inspect
        obj = holder[key]
        try:
            fin = inspect.getattr_static(type(obj), "__del__")
        except AttributeError:
            fin = None
        reader = None if fin is None or isinstance(fin, types.FunctionType) else fin
        for d in (self.census.instances, self.census.fitted):
            for k in [k for k, v in d.items() if v is obj]:
                del d[k]
        obj = fin = None
        if reader is None:   # a Python __del__ is its own call (Census.start / inner: a finalizer)
            del holder[key]
        else:
            self._drop(holder, key, [member], reader)
        gc.collect()

    def _drop(self, holder, key, mids, reader):
        self.census.open_member(sys._getframe(), mids, reader)
        del holder[key]


def _fixture_unit(census, unit, record, done):
    import importlib
    import tempfile
    ml = importlib.import_module("mojolearn")
    fixtures, _, _ = fixture_plan()
    fn = fixtures[unit["id"]][0]
    with tempfile.TemporaryDirectory(prefix="census-fixture-") as tmp:
        cx = FixtureContext(census, tmp)
        ok, error = census.run(lambda: fn(ml, cx))
        record(unit["id"], ok, error)
        done()
        # the fixture's instances the census still holds (probe records), released under the profiler
        held = [census.instances, census.fitted]
        census.instances, census.fitted = {}, {}
        ok, error = census.run(held.clear)
        record("release instances", ok, error)
    done()


def cmd_unit(args):
    inv = json.loads(Path(args.inventory).read_text())
    unit = next((u for u in inv["units"] if u["name"] == args.unit), None)
    if unit is None:
        print(f"unknown unit {args.unit}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (re.sub(r"[^A-Za-z0-9_.-]", "_", unit["name"]) + ".json")
    census = Census(inv)
    fixtures = []
    started = time.time()

    def record(name, ok, error):
        fixtures.append(dict(fixture=name, ok=ok, error=error))

    def done():   # partial results after every fixture, so a crash later in the unit keeps what ran
        doc = dict(unit=unit["name"], seconds=round(time.time() - started, 2), fixtures=fixtures,
                   profiler_errors=dict(census.errors), unattributed=dict(census.unattributed),
                   unattributed_inside=dict(census.unattributed_inside), import_time=dict(census.import_time),
                   members={m: dict(st, funcs=dict(st["funcs"]), bnames=dict(st["bnames"]))
                            for m, st in census.members.items()})
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, sort_keys=True) + "\n")
        os.replace(tmp, path)
    try:
        if unit["kind"] == "api":
            _api_unit(census, unit, record, done, inv["classes"])
        elif unit["kind"] == "light":
            import importlib
            case = getattr(importlib.import_module("mojolearn._verify_light"), unit["id"])
            ok, error = census.run(case)
            record(unit["id"], ok, error)
        elif unit["kind"] == "fixture":
            _fixture_unit(census, unit, record, done)
        else:
            _lane_unit(census, unit, record, done)
    except Exception as exc:
        record("unit", False, f"{type(exc).__name__}: {str(exc)[:300]}")
    done()
    bad = sum(1 for f in fixtures if not f["ok"])
    print(f"{unit['name']}: {len(census.members)} members called, {len(fixtures)} fixtures, {bad} failed, "
          f"profiler errors {dict(census.errors)}", flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------
# report (any python)
# --------------------------------------------------------------------------------------------------------------

def report(directory):
    """Merge the unit results; write census.tsv, functions.tsv, unattributed.tsv; return the SUMMARY lines."""
    d = Path(directory)
    inv = json.loads((d / "inventory.json").read_text())
    merged = {m["id"]: dict(m, stats=_new_stats(), units=set()) for m in inv["members"]}
    unattributed, inside, import_time = collections.Counter(), collections.Counter(), collections.Counter()
    fixture_errors, cpu_refusals, probe_refusals, missing, profiler_errors = 0, 0, 0, [], collections.Counter()
    failures = []   # (unit, fixture, class, error): every failed fixture, by class (failures.tsv)
    for unit in inv["units"]:
        p = d / "units" / (re.sub(r"[^A-Za-z0-9_.-]", "_", unit["name"]) + ".json")
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            missing.append(unit["name"])
            continue
        for f in doc.get("fixtures", ()):
            if not f.get("ok"):
                error = f.get("error") or ""
                # a GPU-only operation refusing on a CPU runner is the product's answer, not a fixture defect
                if any(t in error for t in _CPU_REFUSALS):
                    cpu_refusals += 1
                    kind = "cpu_refusal"
                # a probe ran the member on whatever state the fixtures left (unfitted, closed, a conditional
                # member): its refusal is the product's answer to that state, and the member still ran
                elif f.get("fixture", "").startswith("probe "):
                    probe_refusals += 1
                    kind = "probe_refusal"
                else:
                    fixture_errors += 1
                    kind = "fixture_error"
                failures.append((unit["name"], f.get("fixture", ""), kind, " ".join(error.split())))
        unattributed.update(doc.get("unattributed", {}))
        inside.update(doc.get("unattributed_inside", {}))
        import_time.update(doc.get("import_time", {}))
        profiler_errors.update(doc.get("profiler_errors", {}))
        for mid, st in doc.get("members", {}).items():
            row = merged.setdefault(mid, dict(id=mid, family="?", kind="seen-only", stats=_new_stats(), units=set()))
            acc = row["stats"]
            for k in ("calls", "zero", "frames", "bindings_total", "multi_binding", "no_binding", "exempt", "imports"):
                acc[k] += st.get(k, 0)
            acc["frames_max"] = max(acc["frames_max"], st.get("frames_max", 0))
            acc["bindings_max"] = max(acc["bindings_max"], st.get("bindings_max", 0))
            acc["funcs"].update(st.get("funcs", {}))
            acc["bnames"].update(st.get("bnames", {}))
            row["units"].add(unit["name"])

    needs_gpu, why_not = inv.get("needs_gpu") or {}, inv.get("why_not") or {}
    # a member whose defining function ran through another member (a mixin's method called on a subclass)
    ran = {r.get("defined") for r in merged.values() if r["stats"]["calls"] and r.get("defined")}
    ran |= {r.get("code") for r in merged.values() if r["stats"]["calls"] and r.get("code")}

    def status(row):
        st = row["stats"]
        if st["calls"] == 0:
            if row["kind"] == "native_attribute":
                return "NATIVE_ATTRIBUTE"
            if row.get("family") in PACKAGING_FAMILIES:
                return "PACKAGING"
            if row.get("defined") in ran or (row.get("code") and row["code"] in ran):
                return "VIA_SUBCLASS"
            # a member only a GPU (or a second device) can run, listed with its reason by the census fixtures
            return "NEEDS_GPU" if row["id"] in needs_gpu else "UNCOVERED"
        if st["frames_max"] > 0:
            return "PYTHON"
        return "MULTI_BINDING" if st["bindings_max"] > 1 else ("ZERO" if st["bindings_max"] == 1 else "NO_BINDING")
    rows = sorted(merged.values(), key=lambda r: (r["family"], r["id"]))
    with open(d / "census.tsv", "w") as fh:
        fh.write("member\tfamily\tkind\tstatus\tcalls\tzero_frame_calls\tpy_frames\tpy_frames_max_per_call\t"
                 "distinct_functions\tbinding_calls_max\tbinding_calls_total\texempt_tag_frames\timport_frames\t"
                 "units\ttop_functions\tbindings\n")
        for r in rows:
            st = r["stats"]
            top = "; ".join(f"{k} x{n}" for k, n in st["funcs"].most_common(8))
            bn = "; ".join(f"{k} x{n}" for k, n in st["bnames"].most_common(3))
            units = ",".join(sorted(r["units"])[:4]) + (f",+{len(r['units']) - 4}" if len(r["units"]) > 4 else "")
            fh.write(f"{r['id']}\t{r['family']}\t{r['kind']}\t{status(r)}\t{st['calls']}\t{st['zero']}\t"
                     f"{st['frames']}\t{st['frames_max']}\t{len(st['funcs'])}\t{st['bindings_max']}\t"
                     f"{st['bindings_total']}\t{st['exempt']}\t{st['imports']}\t{units or '-'}\t{top or '-'}\t"
                     f"{bn or '-'}\n")
    by_func = collections.defaultdict(lambda: [0, 0, []])
    for r in rows:
        for k, n in r["stats"]["funcs"].items():
            e = by_func[k]
            e[0] += 1
            e[1] += n
            e[2].append(r["id"])
    funcs = sorted(by_func.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))
    with open(d / "functions.tsv", "w") as fh:
        fh.write("function\tpublic_members\tentries\tmembers\n")
        for k, (nm, ne, ms) in funcs:
            fh.write(f"{k}\t{nm}\t{ne}\t{', '.join(sorted(ms)[:12])}{' ...' if len(ms) > 12 else ''}\n")
    with open(d / "failures.tsv", "w") as fh:
        fh.write("unit\tfixture\tclass\terror\n")
        for row in sorted(failures, key=lambda r: (r[2], r[0], r[1])):
            fh.write("\t".join(row) + "\n")
    with open(d / "unattributed.tsv", "w") as fh:
        # entry: product Python entered outside any public member (a helper the fixture called, a door class the
        # inventory does not list); inside: the frames those entries ran; import_time: product frames run by an
        # import outside any call (connections, meta-path finders)
        fh.write("where\tfunction\tcount\n")
        for where, counter in (("entry", unattributed), ("inside", inside), ("import_time", import_time)):
            for k, n in counter.most_common():
                fh.write(f"{where}\t{k}\t{n}\n")
    # what was not reached, beside the census: UNCOVERED with the fixtures' reason when one is listed, and the
    # NEEDS_GPU list the GPU box's smoke runs (MOJOLEARN_CENSUS_GPU=1)
    with open(d / "uncovered.tsv", "w") as fh:
        fh.write("member\tfamily\tkind\tstatus\treason\n")
        for r in rows:
            st = status(r)
            if st in ("UNCOVERED", "NEEDS_GPU"):
                reason = needs_gpu.get(r["id"]) if st == "NEEDS_GPU" else why_not.get(r["id"], "-")
                fh.write(f"{r['id']}\t{r['family']}\t{r['kind']}\t{st}\t{reason}\n")
    counts = collections.Counter(status(r) for r in rows)
    measured = sum(1 for r in rows if r["stats"]["calls"])
    lines = [f"frames rc=0 measured={measured} with_python={counts['PYTHON']} uncovered={counts['UNCOVERED']} "
             f"needs_gpu={counts['NEEDS_GPU']} (uncovered.tsv) members={len(rows)} "
             f"via_subclass={counts['VIA_SUBCLASS']} "
             f"zero_python={counts['ZERO'] + counts['NO_BINDING'] + counts['MULTI_BINDING']} "
             f"(one_binding_call={counts['ZERO']} multi_binding={counts['MULTI_BINDING']} "
             f"no_binding={counts['NO_BINDING']}) "
             f"native_attributes={counts['NATIVE_ATTRIBUTE']} packaging={counts['PACKAGING']} "
             f"units={len(inv['units'])} "
             f"units_without_results={len(missing)} fixture_errors={fixture_errors} cpu_refusals={cpu_refusals} "
             f"probe_refusals={probe_refusals} (failures.tsv) "
             f"export_errors={len(inv.get('export_errors', ()))} profiler_errors={sum(profiler_errors.values())} "
             f"(reports; gates nothing) census=python-frames/census.tsv"]
    for i, (k, (nm, ne, _)) in enumerate(funcs[:TOP_FUNCTIONS], 1):
        lines.append(f"  frames top{i:02d} members={nm} entries={ne} {k}")
    fams = collections.defaultdict(collections.Counter)
    for r in rows:
        f = fams[r["family"]]
        f["members"] += 1
        f["measured"] += bool(r["stats"]["calls"])
        f[status(r)] += 1
        f["frames"] += r["stats"]["frames"]
    for fam, f in sorted(fams.items(), key=lambda kv: (-kv[1]["PYTHON"], kv[0])):
        lines.append(f"  frames family {fam}: members={f['members']} measured={f['measured']} "
                     f"uncovered={f['UNCOVERED']} zero_python={f['ZERO'] + f['NO_BINDING'] + f['MULTI_BINDING']} "
                     f"with_python={f['PYTHON']} py_frames={f['frames']}")
    if unattributed:
        lines.append(f"  frames unattributed entries (product Python entered outside any public member): "
                     f"{len(unattributed)} distinct, {sum(unattributed.values())} entries; top: "
                     + "; ".join(f"{k} x{n}" for k, n in unattributed.most_common(5)) + " (unattributed.tsv)")
    if import_time:
        lines.append(f"  frames import time (product frames an import ran outside any call, not counted): "
                     f"{sum(import_time.values())} frames; top: "
                     + "; ".join(f"{k} x{n}" for k, n in import_time.most_common(5)))
    exempt = sum(r["stats"]["exempt"] for r in rows)
    lines.append(f"  frames exempt by rule (__sklearn_tags__ and its helpers): {exempt} frames, not counted; "
                 f"import frames (lazy imports inside a call, not counted): {sum(r['stats']['imports'] for r in rows)}")
    if missing:
        lines.append(f"  frames units without results (crashed or timed out before their first fixture): "
                     + ", ".join(missing[:40]) + (f" ... (+{len(missing) - 40})" if len(missing) > 40 else ""))
    return lines


def cmd_report(args):
    for line in report(args.dir):
        print(line)
    return 0


def cmd_run(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import wheel_verify_smoke
    argv = ["frames", "--python", args.python, "--out", args.out]
    if args.jobs:
        argv += ["--jobs", str(args.jobs)]
    return wheel_verify_smoke.main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventory")
    p.add_argument("--surface", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_inventory)
    p = sub.add_parser("unit")
    p.add_argument("--inventory", required=True)
    p.add_argument("--unit", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_unit)
    p = sub.add_parser("report")
    p.add_argument("--dir", required=True)
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("run")
    p.add_argument("--python", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=0)
    p.set_defaults(func=cmd_run)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
