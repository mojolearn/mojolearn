#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FOUR KINDS OF BITWISE-IDENTITY VERIFICATION, PER LANE AND PER SHIPPED ALGORITHM.

    python3 tools/verification_matrix.py            # print the matrix
    python3 tools/verification_matrix.py --write    # rewrite docs/VERIFICATION_MATRIX.md
    python3 tools/verification_matrix.py --check    # fail when the committed doc is stale
    python3 tools/verification_matrix.py --json     # the whole thing as data

Andrew's question, 2026-09-16: "for every algorithm we ship, do we have all
four kinds of bitwise-identity verification?" This tool answers it by READING
THE TREE. Nothing here is typed in by hand; every cell names the file it came
from, and re-running it after a merge produces the new answer rather than a
stale one. It is not a CI gate and is not meant to run per commit. It runs
occasionally, to prove we do what we say.

THE FOUR KINDS

  gpu        a recorded GPU column carries a cell for the lane. Columns are
             admitted by `python/mojolearn/_verify_reference.admit`, the same
             rule the shipped reference table uses: identical mode, a real
             commit, the default fixture size, one device, and NOT a
             sabotage, partial, probe or smoke run. The value is how many of
             the three device classes (apple, nvidia, amd) carry it.
  cpu        a CPU verifier reproduces the GPU bytes for the lane. Declared
             by `python/mojolearn/host_surface.py`, which is the one source
             of the CPU surface: `covered_lanes()` for training and each
             family's `inference_lanes` for prediction from a saved model.
             CORROBORATED when an admitted CPU column also carries the cell.
  sabotage   a negative control that HAS BEEN SEEN to make this lane's bytes
             move. See the next section; this is the cell that is easy to
             fake and the one this tool is most careful about.
  batch      a batch-invariance declaration in `tools/identity_break.py`'s
             BATCH table: either a real part (a function building the calls)
             or an explicit `n/a:<reason>`. An undeclared lane is a gap; the
             harness itself records such a lane as `n/a:UNDECLARED`.

SABOTAGE: "EXISTS" AND "SEEN TO FAIL" ARE DIFFERENT CLAIMS

This tool refuses to report a sabotage as present because a define exists or
because a document says an arm was added. We shipped sabotage that COULD NOT
FAIL more than once (the metrics oracle's old arm, the SVM ties arm, a
Holt-Winters pair whose two flips cancelled, a spectral column that was
trivial), plus a rental guard that matched its own wrapper and a tripwire
written as a rule rather than as code. So the only thing counted as proof
here is a committed pair of columns whose HASHES DIFFER:

  seen(build)    a sabotage-built column (a host binding built with
                 MOJOLEARN_HOST_SABOTAGE or a family's own define, which the
                 column records in `host.families[*].sabotage`, or a build
                 named in the file name) disagrees with a clean column of the
                 same device class, in the same or the parent record
                 directory, on at least one part of this lane's cells.
                 This is the real negative control: the ARITHMETIC moved.
  seen(harness)  the move is only under a harness switch
                 (`batch_sabotage` and its friends, which perturb the
                 harness's own whole-batch evaluation). That proves the batch
                 PROBE can fail. It says nothing about the implementation,
                 so it is reported separately and never counted as a build
                 sabotage.
  declared      the lane's family declares a sabotage define in
                 host_surface.py, but no committed column pair moves this
                 lane. The switch exists. Nobody has watched it fail HERE.
  none          no declared define reaches the lane and no pair moves it.

A sabotage column whose cells REFUSE rather than move is not counted. A
refusal is a build that did not run, not arithmetic that changed.

THE ALGORITHM AXIS, WHICH IS THE POINT

A lane census answers "are our lanes covered". It cannot answer "are our
ALGORITHMS covered", because an algorithm with no lane at all has no row to
be missing from. So the public surface is enumerated independently, from
`__all__` of `python/mojolearn/__init__.py` and of every public submodule it
names, and lanes are mapped ONTO it by walking each lane body's AST for
references rooted at the harness's `ml` handle (`ml.RandomForestClassifier`,
`ml.training.SGD`, `T = ml.training; T.SGD`, and
`from mojolearn.<mod> import <name>`). An algorithm no lane references is
reported first, by name.

An algorithm counts as having a kind when AT LEAST ONE of its lanes has it,
which is the claim "we verify this algorithm that way". Per-lane detail is in
the lane table below it, so a partly covered algorithm is still visible.

WHAT THIS TOOL DOES NOT PROVE. That a lane's fixtures reach every branch, that
a recorded column was honest, that a sabotage arm is a GOOD one (an arm that
moves a lane may still leave the interesting path alone), or anything at all
about FAST. It counts evidence that exists in the tree.
"""
import argparse
import ast
import collections
import glob
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = "docs/VERIFICATION_MATRIX.md"

#: device classes a GPU column can be
GPU_CLASSES = ("apple", "nvidia", "amd")
#: the cell parts a column carries per lane/fixture
PARTS = ("train", "infer", "model", "batch")
#: harness switches: they perturb the harness's own evaluation, not a build
HARNESS_FLAGS = ("batch_sabotage", "rlpair_sabotage", "batchgrad_sabotage",
                 "batchscale_sabotage", "ragged_sabotage")
#: a file name that says "this column is a sabotage build"
SAB_NAME = re.compile(r"sabotage|(^|[-_.])g?sab([-_.]|$)")

#: public names that are not an algorithm: process metadata, tier switches,
#: result containers and option lists. Listed here BY NAME so the exclusion
#: can be argued with rather than hidden in a heuristic.
NOT_ALGORITHMS = frozenset({
    "__version__", "numeric_mode", "set_numeric_mode", "vendor", "gpu_arch",
    "gpu_arch_how", "Array",
    "linalg.numeric_mode", "linalg.require_identical", "linalg.profile",
    "linalg.PROFILE", "linalg.PROFILE_FAMILY", "linalg.PROFILE_VERSION",
    "linalg.PROFILE_BF16", "linalg.PROFILE_INT8", "lowbit.FORMATS",
    "lowbit.BF16Weight", "lowbit.Int8Weight",
    # tokenizer.TrainedBpeVocabulary left this list 2026-09-18
    # (lane/tokenized-corpus): its render_* and write_* produce the two files a
    # user ships beside a model, and tokenizer() builds the encoder, so it is
    # counted, and the bpe-vocabulary lane covers it.
    "training.numeric_mode_used", "training.vendor_used",
    "resample.BootstrapResult", "resample.PermutationTestResult",
    "resample.MonteCarloResult", "resample.STATISTICS", "resample.METHODS",
    "resample.ALTERNATIVES", "resample.INTEGRANDS",
    # Caller-owned state containers (DEVIATION 792), not algorithms. A block
    # RETURNS one from `allocate_state`; nobody computes with it directly. Their
    # buffers are hashed through their block's own lanes and named field by
    # field in the rlpair part's state table in tools/identity_break.py.
    "Mamba1State", "Mamba2State", "Mamba3State", "TransformerState",
    "mamba.Mamba1State", "mamba.Mamba2State", "mamba.Mamba3State",
    "transformer.TransformerState",
    "models.CausalLMState", "models.HFConfig", "models.ModelPlan",
    "models.UnsupportedModel", "models.FAMILIES", "models.PATTERNS",
    "models.causal_lm.CausalLMState", "models.config.HFConfig",
    "models.config.ModelPlan", "models.config.UnsupportedModel",
    "models.config.FAMILIES", "models.config.INTERFACE_DEFAULTS",
    "models.config.FIXED_TODAY", "models.config.POSITION_CEILING",
    "models.tokenizer.PATTERNS", "models.safetensors.TensorInfo",
    "models.safetensors.DTYPES", "models.safetensors.INDEX_NAME",
    "models.safetensors.SINGLE_NAME",
})


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ columns

def record_root(rel):
    """The record directory a column belongs to: the component directly under
    bench/results/<tree>/. A sabotage ARM lives in a subdirectory of it, and
    its clean partner lives at the root, so the two must be told apart."""
    parts = rel.split(os.sep)
    return os.sep.join(parts[:4]) if len(parts) > 4 else os.path.dirname(rel)


def sabotage_signals(j, rel):
    """(is_sabotage, kind). `kind` is 'build' when a BUILD was sabotaged and
    'harness' when only a harness switch was on."""
    flags = [k for k, v in j.items() if k.endswith("_sabotage") and v]
    host_sab = any((f or {}).get("sabotage")
                   for f in ((j.get("host") or {}).get("families") or {}).values()
                   if isinstance(f, dict))
    base = os.path.basename(rel).lower()
    parent = os.path.basename(os.path.dirname(rel)).lower()
    by_name = bool(SAB_NAME.search(base))
    # an arm subdirectory (".../<record>/batch-sabotage/x.json"), never the
    # record directory itself, whose name may mention the lane that made it
    if not by_name and "sabotage" in parent and os.path.dirname(rel) != record_root(rel):
        by_name = True
    if not (flags or host_sab or by_name):
        return False, None
    # A host binding that READS BACK sabotage is a sabotage BUILD, whatever
    # else is on. Otherwise, flags that are all harness switches mean the
    # harness perturbed its own evaluation and no build was touched. A name
    # that says sabotage with neither is a family-define build (the GP
    # gradient, the ARIMA exogenous arm, the GBDT CTR arm), which no generic
    # read-back reports.
    if host_sab:
        return True, "build"
    if flags and all(f in HARNESS_FLAGS for f in flags):
        return True, "harness"
    return True, "build"


def part_value(cell, part):
    """What the column carries for one part of one cell, or None."""
    if not isinstance(cell, dict):
        return None
    vals = cell.get("hashes") if part == "train" else cell.get(part)
    if not vals:
        return None
    return tuple(vals)


def stable_digest(cell, part):
    """A repeated, stable digest; refusal/N/A/one repeat is not evidence."""
    if not isinstance(cell, dict):
        return None
    verdict = cell.get("verdict" if part == "train" else part + "_verdict")
    values = part_value(cell, part)
    if (verdict != "STABLE" or not values or len(values) < 2
            or not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{16}", v) for v in values)
            or len(set(values)) != 1):
        return None
    return values[0]


def negative_control_moves(cell, clean, part):
    """Require a working clean arm and actual changed bytes or a failed assertion."""
    baseline = stable_digest(clean, part)
    if baseline is None:
        return False
    changed = stable_digest(cell, part)
    if changed is not None:
        return changed != baseline
    values = part_value(cell, part)
    verdict = cell.get("verdict" if part == "train" else part + "_verdict", "") if isinstance(cell, dict) else ""
    if verdict not in ("MOVED", "DIVERGENT", "RELOAD-MOVED", "BATCH_MOVED", "RLPAIR_MOVED"):
        return False
    # An explicit assertion failure is evidence, an exception or N/A is not.
    return bool(values and len(values) >= 2 and all(isinstance(v, str) and
        (re.fullmatch(r"[0-9a-f]{16}", v) or v.startswith(("BATCH_MOVED:", "RLPAIR_MOVED:", "RELOAD-MOVED:")))
        for v in values) and any(v != baseline for v in values))


def read_columns(verify_reference):
    """Every identity_break column committed under bench/results/."""
    cols = []
    for path in sorted(glob.glob(os.path.join(ROOT, "bench/results/**/*.json"), recursive=True)):
        try:
            with open(path) as fh:
                j = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(j, dict) or not isinstance(j.get("cells"), dict):
            continue
        rel = os.path.relpath(path, ROOT)
        sab, kind = sabotage_signals(j, rel)
        cols.append(dict(
            rel=rel, dirn=os.path.dirname(rel), root=record_root(rel),
            vendor=j.get("vendor") or "", commit=(j.get("commit") or ""), record=j,
            cls=verify_reference.device_class(j.get("vendor"), path),
            sabotage=sab, sab_kind=kind,
            admit=verify_reference.admit(j, path),
            # A `par-*` driver's claim is only STATEABLE on two devices, and
            # the default rule refuses a two-device column. `par_axis` admits
            # it, and ONLY `par-*` cells may be credited from such a column.
            admit_par=verify_reference.admit(j, path, par_axis=True),
            par_devices=str((j.get("package") or {}).get("par_devices") or "0"),
            cells=j["cells"]))
    return cols


def par_two_device(cols):
    """`par-*` lane -> {device class: record path} from TWO-DEVICE columns.

    The drivers' own axis (2026-09-19). `gpu_coverage` below counts columns
    the DEFAULT rule admits, which refuses `par_devices != "0"` -- correct for
    every ordinary lane and impossible for these, whose claim only exists on
    two devices. Read here through `admit(..., par_axis=True)` instead, and
    only ever for a lane whose name starts `par-`.
    """
    out = collections.defaultdict(dict)
    for c in cols:
        if c["admit_par"] is not None or c["cls"] not in GPU_CLASSES:
            continue
        if c["par_devices"] == "0":
            continue
        for key, cell in c["cells"].items():
            lane = key.split("/", 1)[0]
            if not lane.startswith("par-"):
                continue
            if cell.get("verdict") != "STABLE" or not cell.get("hashes"):
                continue
            out[lane].setdefault(c["cls"], c["rel"])
    return out


def gpu_coverage(cols):
    """lane -> {device class: one record path}, from ADMITTED clean columns."""
    out = collections.defaultdict(dict)
    for c in cols:
        if c["admit"] is not None or c["cls"] not in GPU_CLASSES:
            continue
        for key, cell in c["cells"].items():
            if cell.get("verdict") != "STABLE" or not cell.get("hashes"):
                continue
            out[key.split("/", 1)[0]].setdefault(c["cls"], c["rel"])
    return out


def cpu_recorded(cols):
    """lane -> one admitted CPU column path that carries it."""
    out = {}
    for c in cols:
        if c["admit"] is not None or c["cls"] != "cpu":
            continue
        for key, cell in c["cells"].items():
            if cell.get("verdict") != "STABLE" or not cell.get("hashes"):
                continue
            out.setdefault(key.split("/", 1)[0], c["rel"])
    return out


def sabotage_moves(cols):
    """lane -> {'build': [(part, sabotage path, clean path, same_commit)], ...}

    A sabotage column is paired with a clean column of the SAME DEVICE CLASS,
    preferring the same directory and the same commit, then the parent
    directory, then the record root. A pair whose commits differ is kept but
    marked, because two commits can differ for reasons that are not the
    sabotage."""
    clean = [c for c in cols if not c["sabotage"]]
    moves = collections.defaultdict(lambda: collections.defaultdict(list))
    unpaired = []
    for s in (c for c in cols if c["sabotage"]):
        here = [c for c in clean if c["cls"] == s["cls"] and c["dirn"] == s["dirn"]]
        near = [c for c in clean if c["cls"] == s["cls"]
                and c["dirn"] in (os.path.dirname(s["dirn"]), s["root"])]
        pool = here + near
        same = [c for c in pool if c["commit"] and c["commit"] == s["commit"]]
        partners = same or pool
        if not partners:
            unpaired.append(s["rel"])
            continue
        for key, cell in s["cells"].items():
            lane = key.split("/", 1)[0]
            for part in PARTS:
                for base in partners:
                    other = base["cells"].get(key)
                    if negative_control_moves(cell, other, part):
                        same_commit = bool(s["commit"] and base["commit"] == s["commit"])
                        moves[lane][s["sab_kind"]].append(
                            (part, s["rel"], base["rel"], same_commit))
                        break
    return moves, unpaired


# ------------------------------------------------- the public algorithm surface

def module_all(path):
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError):
        return []
    names = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("__all__", "_DEPRECATED_ALIASES")
                for t in node.targets):
            try:
                names.extend(ast.literal_eval(node.value))
            except (ValueError, TypeError):
                continue
    return list(dict.fromkeys(names))


def package_index():
    """(kinds, aliases) over the package's top level. `aliases` catches
    `LanguageModelTrainer = SmallByteLanguageModelTrainer` and its two peers:
    without it an alias reads as an algorithm with no lane, when the lanes of
    the class it names cover it exactly."""
    kinds, aliases = {}, {}
    for path in sorted(glob.glob(os.path.join(ROOT, "python/mojolearn/*.py"))):
        try:
            tree = ast.parse(Path(path).read_text())
        except (OSError, SyntaxError):
            continue
        rel = os.path.relpath(path, ROOT)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                kinds.setdefault(node.name, ("class", rel))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kinds.setdefault(node.name, ("function", rel))
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name)
                  and isinstance(node.value, ast.Name)):
                aliases.setdefault(node.targets[0].id, node.value.id)
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == "_DEPRECATED_ALIASES"):
                # A renamed class kept importable through a module
                # __getattr__ that warns (tokenizer.py: GPT2Tokenizer ->
                # BpeTokenizer, 2026-09-18): the same kind of alias, spelled
                # as a {old: new} literal because a plain assignment could
                # not warn.
                for old, new in ast.literal_eval(node.value).items():
                    aliases.setdefault(old, new)
    return kinds, aliases


# These modules are public import paths even though they are not re-exported
# by mojolearn.__all__. Omitting them hid the pooled/offloaded trainers and
# parallel drivers from the algorithm axis while their lanes were counted.
EXTRA_PUBLIC_MODULES = (
    "parallel_preprocessing", "parallel_classical", "parallel_ensemble",
    "parallel_graph", "parallel_neighbors", "parallel_neighbors_reference",
    "parallel_training", "model_pool_training", "offload_training",
    "parallel_forecasting",
    "parallel_gaussian_process",
    "parallel_ivf",
    "parallel_model_selection",
)


def package_export(path, export, seen=None):
    """Resolve a nested package export without importing or executing it.

    Resolution stays in its defining module: models.Tokenizer must never be
    assigned the implementation or verification of a same-named root symbol.
    """
    path = Path(path)
    seen = set() if seen is None else seen
    key = (path, export)
    fallback = ("name", os.path.relpath(path, ROOT), export)
    if key in seen or not path.is_file():
        return fallback
    seen.add(key)
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == export:
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            return kind, os.path.relpath(path, ROOT), export
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            for alias in node.names:
                if (alias.asname or alias.name) == export:
                    parent = path.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    target = parent.joinpath(*node.module.split("."))
                    target = target / "__init__.py" if target.is_dir() else target.with_suffix(".py")
                    return package_export(target, alias.name, seen)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == export for t in node.targets):
            if isinstance(node.value, ast.Name):
                return package_export(path, node.value.id, seen)
    return fallback


def public_surface():
    """The shipped surface as {name: (kind, where, target)}, name being a
    top-level export or `<submodule>.<export>`, and `target` the real class
    or function it names (itself, unless it is an alias)."""
    index, aliases = package_index()
    top = module_all(os.path.join(ROOT, "python/mojolearn/__init__.py"))
    surface = {}
    modules = []

    def resolve(export, fallback):
        target = aliases.get(export, export)
        kind, where = index.get(target, ("name", fallback))
        return kind, where, target

    for name in top:
        package = Path(ROOT) / "python/mojolearn" / name
        if (package / "__init__.py").is_file():
            # Public packages have their own export lists and public child
            # modules. Previously `models` was one unresolved name, hiding
            # every loader, tokenizer and whole-model API beneath it.
            for path in sorted(package.rglob("*.py")):
                relative = path.relative_to(package)
                if any(p.startswith("_") for p in relative.parts[:-1]):
                    continue
                if path.name.startswith("_") and path.name != "__init__.py":
                    continue
                parts = relative.parts[:-1] if path.name == "__init__.py" else (*relative.parts[:-1], path.stem)
                module = ".".join((name, *parts))
                modules.append(module)
                for export in module_all(path):
                    surface[f"{module}.{export}"] = package_export(path, export)
            continue
        sub = os.path.join(ROOT, f"python/mojolearn/{name}.py")
        if os.path.exists(sub) and name not in index:
            modules.append(name)
            for export in module_all(sub):
                surface[f"{name}.{export}"] = resolve(export, f"python/mojolearn/{name}.py")
            continue
        surface[name] = resolve(name, "python/mojolearn/__init__.py")
    # `metrics` is `_metrics_impl` imported under that name (__init__.py)
    impl = os.path.join(ROOT, "python/mojolearn/_metrics_impl.py")
    if "metrics" in top and os.path.exists(impl):
        modules.append("metrics")
        surface.pop("metrics", None)
        for node in ast.parse(Path(impl).read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                surface[f"metrics.{node.name}"] = (
                    "function", "python/mojolearn/_metrics_impl.py", node.name)
    for module in EXTRA_PUBLIC_MODULES:
        path = os.path.join(ROOT, f"python/mojolearn/{module}.py")
        if not Path(path).is_file():
            # Older published wheels legitimately predate these modules.
            # Absence belongs in the comparison, not an invented API entry.
            continue
        tree = ast.parse(Path(path).read_text())
        declared = module_all(path)
        modules.append(module)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_") or (declared and node.name not in declared):
                    continue
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                surface[f"{module}.{node.name}"] = (kind, f"python/mojolearn/{module}.py", node.name)
    for m in modules:
        surface.pop(m, None)
    return surface, sorted(set(modules))


def harness_references(harness):
    """Public names `tools/identity_break.py` reaches OUTSIDE any lane body.

    Two paths reach a public name without a lane body naming it, and both are
    real verification:

      the CPU inference routing. `_public_est` swaps the fitted estimator for
      `SambaInference`, `Mamba1BlockInference` and their peers on a CPU
      column, so those classes answer the infer, reload and batch cells of the
      lanes in `NEURAL_PUBLIC_PART_LANES`.

      the opt-in parts. The batchgrad part calls
      `training.accumulate_grads` and `training.accumulation_is_aligned`
      through the same `T = ml.training` alias the lanes use.

    Counting either as "no lane at all" would be wrong, and counting either as
    a lane of its own would be wrong too. They get their own bucket."""
    import inspect
    src = (Path(ROOT) / "tools/identity_break.py").read_text()
    lane_lines = set()
    for fn in harness.LANES.values():
        try:
            lines, start = inspect.getsourcelines(fn)
        except (OSError, TypeError):
            continue
        lane_lines.update(range(start, start + len(lines)))
    tree = ast.parse(src)
    alias = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            v = node.value
            if isinstance(v.value, ast.Name) and v.value.id == "ml":
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        alias[t.id] = v.attr
    refs = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or getattr(node, "lineno", 0) in lane_lines:
            continue
        parts = _ml_chain(node, alias)
        if parts:
            refs |= _chain_refs(parts)
    return refs


def _ml_chain(node, alias):
    """The dotted path of an attribute chain rooted at the harness's `ml`
    (or at a name bound to `ml.<attr>`), or None for a chain rooted
    anywhere else. `ml.models.tokenizer.pattern_name` answers
    `["models", "tokenizer", "pattern_name"]`."""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    if cur.id == "ml":
        root = []
    elif cur.id in alias:
        root = [alias[cur.id]]
    else:
        return None
    parts.reverse()
    return root + parts


def _chain_refs(parts):
    """Every SUFFIX of a dotted path.

    THE HOLE THIS CLOSES (lane/models-namespace-lanes, 2026-09-19). This walk
    used to read two levels: `ml.<a>` answered `<a>` and `ml.<a>.<b>` answered
    `<b>` and `<a>.<b>`, and a third level answered nothing beyond what its
    own prefix answered. The public surface below is enumerated from `__all__`
    of every public submodule INCLUDING a submodule's submodules, so it holds
    eight keys with two dots -- `models.causal_lm.CausalLM`,
    `models.tokenizer.pattern_name` and their six peers -- and no lane body
    could ever have named one of them. All eight were reported "no lane" on
    2026-09-19 while lanes exercised the very objects they name, and no
    spelling of a lane body could have fixed it: `from mojolearn.models...
    import` was the only door that reached a two-dot key, and the harness
    hands lanes an `ml` handle rather than imports. Reporting a gap that
    cannot be closed is the same defect as reporting a pass that cannot fail.

    Suffixes rather than the full path only: the enumeration keys a top-level
    name as `<name>` and a submodule's as `<mod>.<name>`, so a chain must
    answer both. A suffix that matches no enumerated name matches nothing."""
    return {".".join(parts[i:]) for i in range(len(parts))}


def host_family_classes(surface_mod):
    """Public class or function name -> the host families that serve it."""
    out = collections.defaultdict(list)
    for fam in surface_mod.FAMILIES:
        for name in fam.get("classes", ()):
            out[name].append(fam["family"])
    return out


def lane_references(src):
    """Public names one lane body reaches, rooted at the harness's `ml`."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    alias = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            v = node.value
            if isinstance(v.value, ast.Name) and v.value.id == "ml":
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        alias[t.id] = v.attr
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "mojolearn":
            for item in node.names:
                alias[item.asname or item.name] = item.name
    refs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts = _ml_chain(node, alias)
            if parts:
                refs |= _chain_refs(parts)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("mojolearn"):
            sub = node.module.split(".", 1)[1] if "." in node.module else ""
            for a in node.names:
                refs.add(a.name)
                if sub:
                    refs.add(f"{sub}.{a.name}")
    return refs


# ------------------------------------------------------------------- verdicts

_GPU_VACUOUS = None


def gpu_vacuous_lanes():
    """Lanes DEGENERATE on every GPU column, derived (2026-09-19).

    A lane whose arithmetic is the CPU host route, or which stands on no Mojo
    binding at all, runs that box's CPU when handed a GPU column and says
    nothing whatever about the GPU. Both an NVIDIA and an AMD pod printed
    that refusal verbatim today for `cross-val-folds`, `language-model-config`
    and `saved-model-host-infer`, and asking `lane_applicability` directly
    turns up three more: the `byte-lm-host-*` trio.

    So "No GPU column at all" was advertising SIX gaps that no run on any
    hardware can close -- the same unreachable-count defect already fixed for
    `par-*` on the CPU axis and then on the GPU axis. Held out here, and
    reported under their own heading instead.

    Derived by intersecting `lane_applicability.degenerate()` over every GPU
    column, never a list of names: a lane that gains a device path must stop
    being vacuous by itself.
    """
    global _GPU_VACUOUS
    if _GPU_VACUOUS is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import lane_applicability as la
        cols = ("apple-metal", "nvidia-1gpu", "amd-1gpu")
        _GPU_VACUOUS = set.intersection(*(set(la.degenerate(c)) for c in cols))
    return _GPU_VACUOUS


def lane_rows(harness, surface_mod, cols):
    gpu = gpu_coverage(cols)
    two_dev = par_two_device(cols)
    cpu_seen = cpu_recorded(cols)
    moves, unpaired = sabotage_moves(cols)
    covered = set(surface_mod.covered_lanes())
    # A lane that stands on NO host family is still CPU-declared when the
    # host-only registry names it (2026-09-19). `cross-val-folds` is pure
    # Python over the labels and the split count -- no binding, no RNG, no
    # native call -- so no family's `training_lanes` can ever list it, and
    # reading the family registries alone reported it as the one lane with no
    # CPU verifier. It has a CPU recording, an installed CPU-only wheel
    # selects it through `public_reference_lanes()`, and the matrix's own
    # "a GPU column cannot judge" section already says it is checked on the
    # cpu-host column. The count was measuring family membership and calling
    # it verification.
    #
    # MEMBERSHIP ALONE IS NOT THE TEST, and the first version of this made it
    # so. It classified on the registry and carried a comment claiming a
    # host-only lane with no CPU recording would still read as a gap -- which
    # was FALSE: `cpu_recorded` is collected as a field and feeds no count, so
    # nothing would ever have failed. A floor in prose is not a floor. The
    # registry entry now has to be accompanied by an actual committed CPU
    # column, so listing a lane here cannot by itself retire the gap.
    host_only = set(getattr(surface_mod, "PUBLIC_HOST_ONLY_LANES", ()))
    inference = set()
    family_of = {}
    define_of = {}
    for fam in surface_mod.FAMILIES:
        for lane in fam.get("training_lanes", ()):
            family_of.setdefault(lane, fam["family"])
            define_of.setdefault(lane, fam.get("sabotage_define"))
        for lane in fam.get("inference_lanes", ()):
            inference.add(lane)
            family_of.setdefault(lane, fam["family"])
            define_of.setdefault(lane, fam.get("sabotage_define"))

    rows = {}
    for lane in sorted(harness.LANES):
        spec = harness.BATCH.get(lane, "n/a:UNDECLARED")
        if isinstance(spec, str):
            batch = "n/a" if not spec.startswith("n/a:UNDECLARED") else "UNDECLARED"
            batch_why = spec
        else:
            batch = "part"
            batch_why = ""
        m = moves.get(lane, {})
        build = m.get("build", [])
        harness_moves = m.get("harness", [])
        if build:
            sab = "seen(build)"
        elif harness_moves:
            sab = "seen(harness)"
        elif define_of.get(lane):
            sab = "declared"
        else:
            sab = "none"
        cpu_kind = ("training" if lane in covered else
                    "inference" if lane in inference else
                    "host-only" if lane in host_only and cpu_seen.get(lane) else "")
        rows[lane] = dict(
            lane=lane,
            two_device=lane.startswith("par-"),
            gpu_vacuous=lane in gpu_vacuous_lanes(),
            # The drivers' OWN axis: vendor classes carrying a TWO-DEVICE
            # column for this lane. Empty for every non-`par-*` lane.
            par_two=sorted(two_dev.get(lane, {})),
            par_two_where=two_dev.get(lane, {}),
            gpu=sorted(gpu.get(lane, {})),
            gpu_where=gpu.get(lane, {}),
            cpu=cpu_kind,
            cpu_recorded=cpu_seen.get(lane, ""),
            family=family_of.get(lane, ""),
            sabotage=sab,
            sabotage_define=define_of.get(lane) or "",
            sabotage_parts=sorted({p for p, *_ in build}),
            sabotage_where=build[0][1] if build else (harness_moves[0][1] if harness_moves else ""),
            sabotage_same_commit=bool(build and build[0][3]),
            batch=batch, batch_why=batch_why,
            batch_seen=bool([p for p, *_ in (build + harness_moves) if p == "batch"]),
        )
    return rows, unpaired


def has_four(row):
    return (bool(row["gpu"]) and bool(row["cpu"])
            and row["sabotage"] == "seen(build)" and row["batch"] in ("part", "n/a"))


def algorithm_rows(harness, lanes, surface, harness_refs, family_of):
    refs = {}
    import inspect
    def reachable_references(fn, seen):
        # Follow helpers defined in the same harness, such as _rsn. Looking
        # only at a lane body falsely called ReferenceShardedNeighbors
        # uncovered even though two lanes execute it through that helper.
        if fn in seen:
            return set()
        seen.add(fn)
        try:
            source = inspect.getsource(fn)
            import textwrap
            tree = ast.parse(textwrap.dedent(source))
        except (OSError, TypeError, SyntaxError):
            return set()
        found = lane_references(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            helper = fn.__globals__.get(node.func.id)
            if (inspect.isfunction(helper)
                    and helper.__module__ == fn.__module__):
                found.update(reachable_references(helper, seen))
        return found

    for lane, fn in harness.LANES.items():
        refs[lane] = reachable_references(fn, set())
    algos = {}
    for name, (kind, where, target) in sorted(surface.items()):
        if name in NOT_ALGORITHMS or kind == "module":
            continue
        short = name.split(".")[-1]
        qualified = name.split(".")[0] in EXTRA_PUBLIC_MODULES or name.startswith("models.")
        want = {name} if qualified else {name, short, target}
        mine = sorted(l for l, r in refs.items() if want & r)
        rowset = [lanes[l] for l in mine]
        algos[name] = dict(
            name=name, kind=kind, where=where, lanes=mine,
            alias=target if target != short else "",
            routed=bool(want & harness_refs),
            host_family=",".join(family_of.get(short) or family_of.get(target) or []),
            gpu=sorted({c for r in rowset for c in r["gpu"]}),
            cpu=sorted({r["cpu"] for r in rowset if r["cpu"]}),
            sabotage=("seen(build)" if any(r["sabotage"] == "seen(build)" for r in rowset)
                      else "seen(harness)" if any(r["sabotage"] == "seen(harness)" for r in rowset)
                      else "declared" if any(r["sabotage"] == "declared" for r in rowset)
                      else "none" if rowset else ""),
            batch=("part" if any(r["batch"] == "part" for r in rowset)
                   else "n/a" if any(r["batch"] == "n/a" for r in rowset)
                   else "UNDECLARED" if rowset else ""),
            four=bool(rowset) and any(has_four(r) for r in rowset),
        )
    # COLLAPSE PACKAGE RE-EXPORTS (2026-09-19). A public package's
    # `__init__.py` says `from .causal_lm import CausalLM`, so
    # `models.CausalLM` and `models.causal_lm.CausalLM` ARE THE SAME OBJECT.
    # Verified by identity, not inferred: all six `models.*` pairs answer
    # True to `is`. `package_export` already resolves both to one `target`;
    # what was missing is that the ROW was still keyed by name, so one class
    # counted twice.
    #
    # It inflated the public-surface total and, worse, it inflated the GAP.
    # The 2026-09-19 audit read "35 entries with no identity lane", and SIX
    # of those were a second name for an object whose other name sat in the
    # same list. A number that double-counts is not a measure of what is
    # untested, and this one was about to send an agent to write lanes for
    # classes that already had a name in the queue.
    #
    # The kept name is the SHORTEST, which is the one a caller writes
    # (`models.CausalLM`, not `models.causal_lm.CausalLM`); the others are
    # listed in `reexport_of` so the collapse can be audited rather than
    # taken on trust.
    by_target = {}
    for name, row in algos.items():
        key = (row.get("where"), row.get("alias") or name.split(".")[-1])
        by_target.setdefault(key, []).append(name)
    for (where, _), names in by_target.items():
        if len(names) < 2 or not where:
            continue
        # only collapse names inside ONE package, never across the surface
        if len({n.split(".")[0] for n in names}) != 1:
            continue
        keep = min(names, key=lambda n: (n.count("."), len(n)))
        merged = sorted({l for n in names for l in algos[n]["lanes"]})
        algos[keep]["lanes"] = merged
        algos[keep]["reexport_of"] = sorted(n for n in names if n != keep)
        for n in names:
            if n != keep:
                del algos[n]
    return algos


# ------------------------------------------------------------------ rendering

def render(data):
    L, A = data["lanes"], data["algorithms"]
    lanes = [L[k] for k in sorted(L)]
    algos = [A[k] for k in sorted(A)]
    no_lane = [a for a in algos if not a["lanes"] and not a["routed"]]
    routed_only = [a for a in algos if not a["lanes"] and a["routed"]]
    out = []
    w = out.append
    w("# The verification matrix")
    w("")
    w("GENERATED. Do not edit by hand. `python3 tools/verification_matrix.py --write`")
    w("rebuilds it from the tree, and `--check` fails when this file is stale.")
    w("Every number below is read from `tools/identity_break.py`,")
    w("`python/mojolearn/host_surface.py` and the committed columns under")
    w("`bench/results/`. The tool's own docstring says how each cell is decided.")
    w("")
    w("This is a historical coverage inventory, not qualification of the current wheel.")
    w("Public API entries include aliases, wrappers and helpers; they are not a count")
    w("of distinct algorithms. A lane with all four kinds still needs current, matching")
    w("release artifacts and all applicable backend/property checks.")
    w("")
    w("The four kinds, for one lane:")
    w("")
    w("1. **gpu**, a recorded GPU column carries the lane, on 1, 2 or 3 of the")
    w("   device classes apple, nvidia and amd.")
    w("2. **cpu**, a CPU verifier covers the lane, for training or for inference")
    w("   from a saved model, as `host_surface.py` declares it.")
    w("3. **sabotage**, a negative control that HAS BEEN SEEN to move this lane's")
    w("   bytes in a committed pair of columns. `declared` means a define exists")
    w("   and nobody has watched it fail here. `seen(harness)` means only the")
    w("   harness's own batch switch moved, which proves the probe can fail and")
    w("   says nothing about the implementation.")
    w("4. **batch**, a declared batch-invariance part, or a named `n/a:<reason>`.")
    w("")
    w("## The numbers")
    w("")
    w(f"- Lanes: **{len(lanes)}** ({sum(1 for r in lanes if not r['two_device'])} single-device, "
      f"{sum(1 for r in lanes if r['two_device'])} `par-*` multi-GPU drivers).")
    w(f"- Source public API entries enumerated from the public API: **{len(algos)}**.")
    w(f"- Source public API entries with ALL FOUR kinds on at least one lane: "
      f"**{sum(1 for a in algos if a['four'])}** of {len(algos)}.")
    w(f"- Source public API entries with NO IDENTITY LANE AT ALL: **{len(no_lane)}**.")
    w(f"- Source public API entries with no lane of their own, but reached by the harness's")
    w(f"  CPU inference routing: **{len(routed_only)}**.")
    w("")
    w("Per kind, over the public API entries:")
    w("")
    w("| kind | API entries that have it | missing |")
    w("|---|---|---|")
    for label, key in (("gpu column", "gpu"), ("cpu verifier", "cpu"),
                       ("sabotage seen to move a build", "sabotage"),
                       ("batch part or named n/a", "batch")):
        if key == "sabotage":
            have = sum(1 for a in algos if a["sabotage"] == "seen(build)")
        elif key == "batch":
            have = sum(1 for a in algos if a["batch"] in ("part", "n/a"))
        else:
            have = sum(1 for a in algos if a[key])
        w(f"| {label} | {have} | {len(algos) - have} |")
    w("")
    w("Per kind, over the lanes:")
    w("")
    w("| kind | lanes that have it | missing |")
    w("|---|---|---|")
    g3 = sum(1 for r in lanes if len(r["gpu"]) == 3)
    w(f"| gpu column (any class) | {sum(1 for r in lanes if r['gpu'])} | "
      f"{sum(1 for r in lanes if not r['gpu'])} |")
    w(f"| gpu column on all three classes | {g3} | {len(lanes) - g3} |")
    w(f"| cpu verifier declared | {sum(1 for r in lanes if r['cpu'])} | "
      f"{sum(1 for r in lanes if not r['cpu'])} |")
    w(f"| sabotage seen to move a build | {sum(1 for r in lanes if r['sabotage'] == 'seen(build)')} | "
      f"{sum(1 for r in lanes if r['sabotage'] != 'seen(build)')} |")
    w(f"| batch part or named n/a | {sum(1 for r in lanes if r['batch'] in ('part', 'n/a'))} | "
      f"{sum(1 for r in lanes if r['batch'] not in ('part', 'n/a'))} |")
    w(f"| ALL FOUR | {sum(1 for r in lanes if has_four(r))} | "
      f"{sum(1 for r in lanes if not has_four(r))} |")
    w("")
    w("Sabotage, split by what was actually watched:")
    w("")
    w("| verdict | lanes | what it means |")
    w("|---|---|---|")
    for v, meaning in (("seen(build)", "a sabotage BUILD moved the bytes; a real negative control"),
                       ("seen(harness)", "only the harness batch switch moved; the probe can fail, the build is unproven"),
                       ("declared", "the family declares a define; no committed pair moves this lane"),
                       ("none", "no define reaches the lane and nothing has moved it")):
        w(f"| {v} | {sum(1 for r in lanes if r['sabotage'] == v)} | {meaning} |")
    w("")

    w("## Source public API entries with no identity lane at all")
    w("")
    if not no_lane:
        w("None.")
    else:
        w("These are the most important gaps. An algorithm with no lane cannot")
        w("be missing a cell, so a lane census hides it entirely. `host family`")
        w("names the CPU host family that serves the class, where one does, which")
        w("means a CPU path exists and only the identity lane is missing.")
        w("")
        w("| algorithm | kind | host family | defined in |")
        w("|---|---|---|---|")
        for a in no_lane:
            w(f"| `{a['name']}` | {a['kind']} | {a['host_family'] or '-'} | `{a['where']}` |")
        w("")
        w("THE SAVED-MODEL HOST INFERENCE SURFACE used to be listed here with the")
        w("note that it had no lane ON PURPOSE, because `tools/forest_host_gate.py`")
        w("and `tools/classical_host_gate.py` measure it against committed")
        w(f"recordings instead: {data['forest_recordings']} under")
        w(f"`bench/results/forest_host/` and {data['classical_recordings']} classical")
        w("recording directories named in `host_surface.py`. Those gates are real")
        w("and they pass. What they did not reach (lane/laneless-public-classes,")
        w("2026-09-19) is `host_predict` and `host_predict_proba`, which no gate")
        w("calls, and the `parallel_groves` HOST engine, which the forest gate still")
        w("refuses by name although `core/forest_host_groves.mojo` landed on")
        w("lane/forest-groves-cpu-and-speed. The `saved-model-host-infer` lane runs")
        w("all three, so the surface is counted in the four kinds below; the gates")
        w("remain a different and additional kind of evidence.")
    w("")
    w("### Reached by the harness, with no lane of their own")
    w("")
    w("`tools/identity_break.py` reaches these outside any lane body. `_public_est`")
    w("swaps the fitted estimator for the public CPU inference class on a CPU")
    w("column, so those answer the infer, reload and batch cells of the lanes named")
    w("in `NEURAL_PUBLIC_PART_LANES` (" + ", ".join(data["neural_public_lanes"]) + ");")
    w("the batchgrad part calls the accumulation helpers the same way.")
    w("They are verified, but no lane carries their name.")
    w("")
    if not routed_only:
        w("None.")
    else:
        w("| algorithm | kind | host family | defined in |")
        w("|---|---|---|---|")
        for a in routed_only:
            w(f"| `{a['name']}` | {a['kind']} | {a['host_family'] or '-'} | `{a['where']}` |")
    w("")

    w("## The algorithm matrix")
    w("")
    w("`gpu` is the device classes that carry any of the algorithm's lanes.")
    w("A blank cell means no lane of this algorithm has that kind.")
    w("")
    w("| algorithm | lanes | gpu | cpu | sabotage | batch | all four |")
    w("|---|---|---|---|---|---|---|")
    for a in algos:
        lane_txt = str(len(a["lanes"])) if a["lanes"] else ("routed" if a["routed"] else "**0**")
        name = f"`{a['name']}`" + (f" (alias of `{a['alias']}`)" if a["alias"] else "")
        w(f"| {name} | {lane_txt} | {','.join(a['gpu'])} | {','.join(a['cpu'])} | "
          f"{a['sabotage']} | {a['batch']} | {'yes' if a['four'] else 'NO'} |")
    w("")

    w("## The lane matrix")
    w("")
    w("| lane | gpu | cpu | sabotage | sabotage evidence | batch | all four |")
    w("|---|---|---|---|---|---|---|")
    for r in lanes:
        ev = r["sabotage_where"]
        if ev and not r["sabotage_same_commit"] and r["sabotage"] == "seen(build)":
            ev += " (clean partner at another commit)"
        w(f"| {r['lane']} | {','.join(r['gpu']) or '-'} | {r['cpu'] or '-'} | {r['sabotage']} | "
          f"{'`' + ev + '`' if ev else '-'} | {r['batch']}{(' ' + r['batch_why']) if r['batch'] == 'n/a' else ''} | "
          f"{'yes' if has_four(r) else 'NO'} |")
    w("")

    w("## Lanes missing each kind, by name")
    w("")
    w("A `par-*` lane IS NOT A CPU GAP AND NEVER WILL BE. It is a multi-device")
    w("driver whose whole claim, written in `identity_break._par_devices`, is")
    w("that a TWO-DEVICE column hashes equal to the one-device column cell for")
    w("cell. A CPU column has zero devices, so that claim there is not false,")
    w("it is NOT EXPRESSIBLE -- and an inexpressible claim listed beside real")
    w("gaps is noise that hides them. On 2026-09-19 it hid them at a ratio of")
    w("36 to 3. These lanes need a SECOND GPU, not another CPU run, and no CPU")
    w("work will ever close them; they are counted below under their own")
    w("heading and excluded from the two CPU-axis gap lists by construction.")
    w("")
    w("AND A `par-*` LANE IS NOT A VENDOR-CLASS GAP EITHER (2026-09-19). The")
    w("rules make that count UNREACHABLE for them, in both directions at once:")
    w("")
    w("  * `_verify_reference.py:314` REFUSES any column recording")
    w("    `par_devices != \"0\"`, and `gpu_coverage` above counts ADMITTED")
    w("    columns only -- so a two-device column is invisible to this number")
    w("    BY CONSTRUCTION;")
    w("  * and on ONE device every one of them is DEGENERATE, measured:")
    w("    `lane_applicability.degenerate('apple-metal')` holds all 13 of them,")
    w("    because with one shard the equality they assert is not false, it is")
    w("    not expressible.")
    w("")
    w("So the claim is only STATEABLE on two devices and only ADMISSIBLE on")
    w("one. No run, on any hardware, ever, can take a `par-*` lane to three")
    w("vendor classes under these rules, and listing them as short of it has")
    w("been advertising 13 gaps that cannot be closed. A $3.34 two-device")
    w("MI300X leg was bought on 2026-09-19 before this was noticed; what it")
    w("proved is real and is reported under the driver heading below, not here.")
    w("")
    for label, pred, cpu_axis, gpu_axis in (
            ("No GPU column at all", lambda r: not r["gpu"], False, True),
            ("GPU column on fewer than three classes", lambda r: 0 < len(r["gpu"]) < 3, False, True),
            ("No CPU verifier declared", lambda r: not r["cpu"], True, False),
            ("Sabotage not seen to move a build", lambda r: r["sabotage"] != "seen(build)", True, False),
            ("Batch undeclared", lambda r: r["batch"] == "UNDECLARED", False, False)):
        hit = [r for r in lanes if pred(r)
               and not ((cpu_axis or gpu_axis) and r["two_device"])
               and not (gpu_axis and r.get("gpu_vacuous"))]
        w(f"**{label}: {len(hit)}**")
        w("")
        w("> " + (", ".join(r["lane"] for r in hit) if hit else "none"))
        w("")
        if cpu_axis or gpu_axis:
            held = [r for r in lanes if pred(r) and r["two_device"]]
            why = ("a CPU column cannot state their claim" if cpu_axis else
                   "this count is unreachable for them in both directions")
            w(f"> (plus {len(held)} `par-*` multi-GPU driver lanes, held out of "
              f"this count: {why}. They are listed once below.)")
            w("")

    # `par-*` are vacuous on a one-device GPU column too, but they have
    # their own heading below and the reason there is different (two
    # devices, not a host route). Listed once, under the right one.
    vac = [r for r in lanes if r.get("gpu_vacuous") and not r["two_device"]]
    if vac:
        w("## Lanes a GPU column cannot judge at all")
        w("")
        w(f"{len(vac)} lanes are DEGENERATE on every GPU column. Their arithmetic is")
        w("the CPU host route, or they stand on no Mojo binding at all, so handed")
        w("a GPU column they run that box's CPU and say nothing whatever about the")
        w("GPU. An NVIDIA and an AMD pod each printed that refusal verbatim on")
        w("2026-09-19; `lane_applicability` names three more. They are held out of")
        w("the two GPU-axis counts above because listing them there advertised six")
        w("gaps no run on any hardware can close -- the same unreachable count")
        w("already fixed for `par-*`. THEY ARE NOT UNVERIFIED: each is checked on")
        w("the cpu-host column, which is the one column its proposition is")
        w("stateable on.")
        w("")
        for r in sorted(vac, key=lambda x: x["lane"]):
            w(f"| {r['lane']} | cpu: {r['cpu'] or '-'} | sabotage: {r['sabotage']} |")
        w("")

    drivers = [r for r in lanes if r["two_device"]]
    waiting = [r for r in drivers if r["sabotage"] != "seen(build)" or not r["cpu"]]
    w("## The multi-GPU driver lanes, which a CPU column cannot judge")
    w("")
    have_two = [r for r in drivers if r.get("par_two")]
    w(f"{len(drivers)} `par-*` lanes exist. THEIR CLAIM IS ONLY STATEABLE ON "
      "TWO DEVICES -- that a two-device column hashes equal to the one-device "
      "column cell for cell -- so a one-device run of one is DEGENERATE: it "
      "compares a run against itself and passes whatever the code does. They "
      "are held out of the vendor-class counts above for that reason.")
    w("")
    w(f"**{len(have_two)} of {len(drivers)} now carry a TWO-DEVICE column**, read "
      "through `admit(..., par_axis=True)`. Until 2026-09-19 the default rule "
      "refused `par_devices != \"0\"`, so the only run that can state their "
      "claim was inadmissible and this evidence counted for nothing.")
    w("")
    for r in sorted(drivers, key=lambda x: x["lane"]):
        got = ",".join(r.get("par_two") or []) or "-"
        w(f"| {r['lane']} | {got} | {next(iter((r.get('par_two_where') or {}).values()), '-')} |"
          if r.get("par_two") else f"| {r['lane']} | - | no two-device column |")
    w("")
    w("> " + (", ".join(r["lane"] for r in waiting) if waiting else "none"))
    w("")

    w("## Public names not counted as algorithms")
    w("")
    w("Listed by name in `NOT_ALGORITHMS` in the tool, so the exclusion can be")
    w("argued with rather than hidden in a heuristic. These are process metadata,")
    w("tier switches, result containers, option lists and the caller-owned state")
    w("containers a block returns from `allocate_state`, whose buffers are hashed")
    w("through their block's own lanes.")
    w("")
    w("> " + ", ".join(f"`{n}`" for n in data["not_algorithms"]))
    w("")
    if data["unpaired"]:
        w("## Sabotage columns this tool could not pair")
        w("")
        w("A sabotage column with no clean partner of the same device class in")
        w("its own, parent or record directory. Its evidence is not counted.")
        w("")
        for p in data["unpaired"]:
            w(f"- `{p}`")
        w("")
    return "\n".join(out) + "\n"


def build():
    harness = load("tools/identity_break.py", "_vm_identity_break")
    surface_mod = load("python/mojolearn/host_surface.py", "_vm_host_surface")
    verify_reference = load("python/mojolearn/_verify_reference.py", "_vm_verify_reference")
    cols = read_columns(verify_reference)
    lanes, unpaired = lane_rows(harness, surface_mod, cols)
    surface, modules = public_surface()
    algos = algorithm_rows(harness, lanes, surface,
                           harness_references(harness),
                           host_family_classes(surface_mod))
    forest_root = os.path.join(ROOT, getattr(surface_mod, "FOREST_RECORDED_ROOT", ""))
    return dict(lanes=lanes, algorithms=algos, unpaired=unpaired,
                columns=len(cols), modules=modules,
                not_algorithms=sorted(NOT_ALGORITHMS),
                forest_recordings=len(glob.glob(os.path.join(forest_root, "*"))),
                classical_recordings=len(getattr(surface_mod, "CLASSICAL_RECORDED", ())),
                neural_public_lanes=list(getattr(harness, "NEURAL_PUBLIC_PART_LANES", ())))


def unreachable_gaps(lanes):
    """Gap-list entries that NO RUN ON ANY HARDWARE could close.

    FOUR TIMES ON 2026-09-19 a heading in this document advertised work
    nobody could do, which is worse than silence because somebody rents a box
    for it:

      * `par-*` on the CPU axis -- 36 inexpressible claims sitting on top of
        3 real ones, a 12:1 noise ratio that hid them for a day;
      * `par-*` on the GPU axis -- 13 lanes whose claim is STATEABLE only on
        two devices and ADMISSIBLE only on one, so the count was unreachable
        in both directions at once;
      * six host-routed lanes under "No GPU column at all", whose arithmetic
        is the CPU route and which measure that box's CPU when handed a GPU;
      * `admit` refusing `par_devices != "0"`, which threw away the only
        column able to state the drivers' proposition -- 50 of 51 lanes had
        two-device proof sitting unread for five days.

    Each was found by hand, late, after the number had already sent work
    somewhere. This is the check for the fifth: a lane may only be listed as
    missing a vendor class if it can actually STATE its proposition on that
    class's column. `lane_applicability.degenerate()` is the same rule
    `verify_lanes.py` enforces at run time, asked here at report time.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lane_applicability as la
    col_for = {"apple": "apple-metal", "nvidia": "nvidia-1gpu", "amd": "amd-1gpu"}
    deg = {cls: set(la.degenerate(col)) for cls, col in col_for.items()}
    bad = []
    rows = lanes.values() if isinstance(lanes, dict) else lanes
    for r in rows:
        if r["two_device"] or r.get("gpu_vacuous"):
            continue                      # already held out, with a reason
        for cls in col_for:
            if cls not in (r["gpu"] or []) and r["lane"] in deg[cls]:
                bad.append((r["lane"], cls))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true", help=f"rewrite {DOC}")
    ap.add_argument("--check", action="store_true", help=f"fail when {DOC} is stale")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    data = build()
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True, default=list))
        return 0
    text = render(data)
    path = os.path.join(ROOT, DOC)
    if args.write:
        with open(path, "w") as fh:
            fh.write(text)
        print(f"wrote {DOC}: {len(data['lanes'])} lanes, {len(data['algorithms'])} public API entries, "
              f"{data['columns']} columns read")
        return 0
    if args.check:
        try:
            have = Path(path).read_text()
        except OSError:
            print(f"verification_matrix: {DOC} is missing; run --write", file=sys.stderr)
            return 1
        if have != text:
            print(f"verification_matrix: {DOC} is STALE; run --write", file=sys.stderr)
            return 1
        bad = unreachable_gaps(data["lanes"])
        if bad:
            print("verification_matrix: a gap list names work NO RUN CAN DO -- "
                  "the lane is degenerate on the very column it is said to lack:",
                  file=sys.stderr)
            for lane, cls in bad:
                print(f"  {lane} listed as missing {cls}, but is degenerate there",
                      file=sys.stderr)
            print("  Hold them out with a reason, as par-* and the host-routed "
                  "lanes already are.", file=sys.stderr)
            return 1
        print(f"verification_matrix OK: {DOC} matches the tree "
              f"({len(data['lanes'])} lanes, {len(data['algorithms'])} public API entries); "
              "no gap list names unreachable work")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
