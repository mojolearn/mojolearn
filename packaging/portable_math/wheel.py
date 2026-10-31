#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Finalize and audit wheels without platform math imports (build-only LIEF)."""
import argparse
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile

from stage import stage

# The IDENTICAL CUDA PTX rounding audit lives beside the Linux set builder
# that applies it (packaging/linux/build_sets.sh).
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linux"))
import ptx_contract  # noqa: E402
import cubin_contract  # noqa: E402
import ptx_baseline  # noqa: E402

# C99 math entry points, including float/long-double variants. A new runtime
# dependency must fail closed instead of being silently removed by the patcher.
ROOTS = "acos acosh asin asinh atan atan2 atanh cbrt ceil copysign cos cosh erf erfc exp exp2 expm1 fabs fdim floor fma fmax fmin fmod frexp hypot ilogb ldexp lgamma llrint llround log log10 log1p log2 logb lrint lround modf nearbyint nextafter nexttoward pow remainder remquo rint round scalbln scalbn sin sincos sinh sqrt tan tanh tgamma trunc".split()
MATH_SYMBOLS = {root + suffix for root in ROOTS for suffix in ("", "f", "l")}
# Apple's compiler lowers sin+cos pairs (and sinpi/cospi) to these private
# libSystem entry points; 0.8.19's macOS audit missed __sincosf_stret.
MATH_SYMBOLS |= {"__sincosf_stret", "__sincos_stret", "__sincospif_stret", "__sincospi_stret",
                 "__sinpif", "__cospif", "__tanpif", "__sinpi", "__cospi", "__tanpi",
                 "__exp10f", "__exp10"}
#: A FAST-tier binding of a Linux GPU set: mojolearn/<cuda|hip>/<arch>/<name>.so
#: with no tier directory (identical/ sits one level deeper). The merged
#: bindings' native file of a FAST set (`_mojolearn_native.so`, holding every
#: FAST member, packaging/linux/merged_layout.py) matches the same pattern;
#: its index `_mojolearn_merged.json` is not a binary and is never audited here.
FAST_SET = re.compile(r"^mojolearn/(cuda|hip|cuda_native|hip_native|cuda_ptx)/[^/]+/_mojolearn[^/]*\.so$"
                      # macOS: FAST GPU bindings sit at the package root; host/
                      # and identical/ are directories below it.
                      r"|^mojolearn/_mojolearn[^/]*\.so$")
# These entry points run independent tests; none is imported by estimators.
NUMPY_ORACLES = {
    "mojolearn/_identity_break.py", "mojolearn/_identity.py",
    "mojolearn/_verify_all.py", "mojolearn/_verify_distributed.py",
    "mojolearn/_verify_parallel_cv.py", "mojolearn/_parallel_cv_witness.py",
    # Exact testing controllers registered by the verify-* CLI. Their NumPy
    # imports are lazy fixture/oracle/report work; product paths never call
    # them. Keep exact names: a _verify_* prefix is not a runtime exemption.
    "mojolearn/_verify_distributed_program.py", "mojolearn/_verify_generation.py",
    "mojolearn/_verify_graph.py", "mojolearn/_verify_partitioned_program.py",
    # The installed public-API verifier was split into these exact controllers.
    # They construct reference fixtures/assertions, never product results.
    "mojolearn/_verify_api.py", "mojolearn/_verify_api_estimators.py",
    "mojolearn/_verify_api_faults.py", "mojolearn/_verify_api_metadata.py",
    "mojolearn/_verify_api_misc.py", "mojolearn/_verify_api_models.py",
    "mojolearn/_verify_api_neural.py", "mojolearn/_verify_api_optimizers.py",
    "mojolearn/_verify_api_programs.py", "mojolearn/_verify_api_runner.py",
    "mojolearn/_verify_api_training.py", "mojolearn/_verify_api_utilities.py",
    "mojolearn/_verify_light.py",
}


def native_setup_numpy_imports(tree, relative):
    """Exact optional import statements which connect native metadata at setup.

    Do not exempt a module, an arbitrary function, or an unguarded dependency.
    Product computation remains audited; these imports populate the retained
    native type/descriptor tables once during package/reader installation.
    """
    setup = {
        "mojolearn/_expansion_neighbors.py": "_connect_pcs_rng",
        "mojolearn/_native_xcnn_bridge.py": "connect",
        "mojolearn/importers/_native_reader_bridge.py": "connect",
        "mojolearn/resample.py": None,
    }
    if relative not in setup:
        return set()
    scopes = [tree] if setup[relative] is None else [node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == setup[relative]]
    admitted = set()
    # Walk compound statements, but never enter a nested callable/class: its
    # execution may be a later user operation rather than import setup.
    def statements(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            yield node
            for field in ("body", "orelse", "finalbody"):
                yield from statements(getattr(node, field, []))
    for scope in scopes:
        for node in statements(scope.body):
            if not isinstance(node, ast.Try):
                continue
            if not any(isinstance(handler.type, ast.Name) and handler.type.id in
                       ("ImportError", "ModuleNotFoundError") for handler in node.handlers):
                continue
            for statement in node.body:
                if isinstance(statement, ast.Import) and len(statement.names) == 1 and statement.names[0].name == "numpy":
                    admitted.add(id(statement))
                elif (relative == "mojolearn/_expansion_neighbors.py" and isinstance(statement, ast.ImportFrom)
                      and statement.level == 0 and statement.module == "numpy.random"
                      and {name.name for name in statement.names} ==
                      {"mtrand", "BitGenerator", "MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"}):
                    admitted.add(id(statement))
    return admitted


def numpy_errors(path, relative):
    errors = []
    if any(re.match(r"numpy(?:$|\.py$|\.libs$|-[^/]+\.dist-info$)", part, re.I) for part in Path(relative).parts):
        errors.append(relative + ": bundled NumPy payload")
    if path.name == "METADATA":
        from email.parser import Parser
        metadata = Parser().parsestr(path.read_text())
        if any(re.match(r"numpy(?:$|[\s<>=!~;\[])", dep, re.I)
               and not any(extra in metadata.get_all("Provides-Extra", [])
                           and re.fullmatch(r"\s*extra\s*==\s*[\"']" + extra + r"[\"']\s*", dep.partition(';')[2])
                           for extra in ("numpy", "verify"))
               for dep in metadata.get_all("Requires-Dist", [])):
            errors.append(relative + ": NumPy dependency metadata")
    if path.suffix == ".py" and relative not in NUMPY_ORACLES:
        tree = ast.parse(path.read_bytes(), filename=relative)
        optional_imports = native_setup_numpy_imports(tree, relative)
        # Optional runtime support is separate from the independent verifier.
        # Only this lazy, actionable dependency guard may import NumPy for APIs.
        if relative == "mojolearn/_optional_numpy.py":
            for fn in tree.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "require_numpy":
                    optional_imports.update(id(n) for n in ast.walk(fn) if isinstance(n, ast.Import))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                func = node.func
                if (isinstance(func, ast.Name) and func.id == "__import__"
                        or isinstance(func, ast.Attribute) and func.attr == "import_module"):
                    names.append(str(node.args[0].value))
            if any(name.split(".")[0] == "numpy" for name in names) and id(node) not in optional_imports:
                errors.append(relative + ": NumPy import outside verification or optional runtime guard")
    return errors


# THE WHEEL DEPENDS ON NOTHING BUT PYTHON AND THE MOJO/MAX RUNTIME IT BUNDLES.
# `dependencies = []` in pyproject.toml says so; this makes a shipped module that
# imports a third-party package fail the build instead of failing on a user's box.
# Module level: the standard library and mojolearn only. Inside a function: also
# an optional interop package, which the caller must tolerate being absent.
OPTIONAL_LAZY_IMPORTS = {"sklearn"}  # scikit-learn protocol hooks; each falls back when it is missing
# A source-tree-only check the shipped copy of tools/identity_break.py skips
# by name when tools/ is absent (it reads bindings/*.mojo, which never ship).
SOURCE_TREE_LAZY_IMPORTS = {"mojolearn/_identity_break.py": {"lane_applicability"}}
# The packages a pretrained model is rebuilt from, imported only inside the catalog check that rebuilds it
# (mojolearn.models.verify_catalog_entry): the user who checks a torchvision, Hugging Face or ONNX model has its source
# package; every other path of the wheel never imports them.
MODEL_SOURCE_LAZY_IMPORTS = {"mojolearn/models/_catalog.py": {"torch", "torchvision", "transformers", "onnx"}}
# The framework adapters (packaging/adapters/): each module imports its own framework, inside the function
# that refuses with an install hint when it is missing; `import mojolearn` never reaches either.
# mojolearn.import_model imports PyTorch the same way (importers/_program.py `_require_torch`): only when the user
# passes a torch.nn.Module, and its ImportError names `pip install torch`.
ADAPTER_LAZY_IMPORTS = {"mojolearn/adapters/torch.py": {"torch"}, "mojolearn/adapters/_torch_compile.py": {"torch"},
                        "mojolearn/adapters/max.py": {"max"}, "mojolearn/importers/_program.py": {"torch"}}


def sparse_output_adapter_imports(tree, relative):
    """The caller supplied sparse data; only its CSR return adapter needs SciPy.

    This is deliberately narrower than allowing lazy SciPy across the package:
    dense paths and all arithmetic must remain independent of SciPy.
    """
    if relative != "mojolearn/_expansion_neighbors.py":
        return set()
    allowed = set()
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != "AdditiveChi2Sampler":
            continue
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name != "transform":
                continue
            for branch in fn.body:
                if not isinstance(branch, ast.If):
                    continue
                guard = branch.test
                if not (isinstance(guard, ast.Compare) and isinstance(guard.left, ast.Name)
                        and guard.left.id == "sparse" and len(guard.ops) == 1
                        and isinstance(guard.ops[0], ast.IsNot) and len(guard.comparators) == 1
                        and isinstance(guard.comparators[0], ast.Constant)
                        and guard.comparators[0].value is None):
                    continue
                for node in branch.body:
                    if isinstance(node, ast.Import) and all(a.name == "scipy.sparse" for a in node.names):
                        allowed.add(id(node))
    return allowed


def dependency_errors(path, relative):
    """A shipped .py importing anything but the standard library and mojolearn
    (NumPy is judged by numpy_errors; an optional interop package only lazily)."""
    if path.suffix != ".py":
        return []
    import sys
    allowed = set(sys.stdlib_module_names) | {"mojolearn", "numpy", "__future__"}
    errors = []
    tree = ast.parse(path.read_bytes(), filename=relative)
    sparse_adapters = sparse_output_adapter_imports(tree, relative)
    # Python 3.14's optional stdlib zstd reader is guarded on older supported
    # interpreters. The audit interpreter's stdlib list must not turn that
    # explicit ImportError fallback into a missing third-party dependency.
    optional_stdlib = set()
    for block in ast.walk(tree):
        if isinstance(block, ast.Try) and any(isinstance(h.type, ast.Name)
                and h.type.id == 'ImportError' for h in block.handlers):
            for statement in block.body:
                if isinstance(statement, ast.ImportFrom) and statement.module == 'compression':
                    optional_stdlib.add(id(statement))
    lazy = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            lazy.update(id(n) for n in ast.walk(fn))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top in allowed or top.startswith("_mojolearn"):
                continue
            if top == 'compression' and id(node) in optional_stdlib:
                continue
            if top in OPTIONAL_LAZY_IMPORTS and id(node) in lazy:
                continue
            if id(node) in sparse_adapters:
                continue
            if top in SOURCE_TREE_LAZY_IMPORTS.get(relative, ()) and id(node) in lazy:
                continue
            if top in MODEL_SOURCE_LAZY_IMPORTS.get(relative, ()) and id(node) in lazy:
                continue
            if top in ADAPTER_LAZY_IMPORTS.get(relative, ()) and id(node) in lazy:
                continue
            errors.append(f"{relative}: imports {top!r}, which the wheel does not provide")
    return errors


LIBM = re.compile(r"^lib(?:m|mvec)(?:[.-]|$)", re.I)


# The native module registry (mojolearn/_native_loaded_modules.py) connects the
# `math` module so compiled bindings can read a few exact functions from it.
# libm transcendental math (sin, exp, sqrt, log, pow, ...) is platform-dependent
# and forbidden in the runtime; rounding to an integer, exact binomials, gcd and
# the finite/NaN/inf predicates and constants are exact on every platform, so
# they are not arithmetic the platform can vary. Only these names may be
# connected; any other export name fails the audit.
MATH_EXACT_EXPORTS = frozenset({"floor", "ceil", "comb", "trunc", "gcd", "isfinite", "isnan", "isinf", "inf", "nan"})


def exact_math_connection_imports(tree):
    """Admit the registry's `import math` only when every use of the name
    `math` is the module argument of one `connect_module('math', math, (...))`
    call whose literal export tuple is a subset of MATH_EXACT_EXPORTS."""
    imports = [node for node in tree.body if isinstance(node, ast.Import) and len(node.names) == 1
               and node.names[0].name == "math" and node.names[0].asname is None]
    if len(imports) != 1:
        return set()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "connect_module" and len(node.args) == 3 and not node.keywords
             and isinstance(node.args[0], ast.Constant) and node.args[0].value == "math"]
    if len(calls) != 1:
        return set()
    call = calls[0]
    module, exports = call.args[1], call.args[2]
    if not (isinstance(module, ast.Name) and module.id == "math" and isinstance(exports, ast.Tuple)
            and all(isinstance(e, ast.Constant) and type(e.value) is str for e in exports.elts)
            and {e.value for e in exports.elts} <= MATH_EXACT_EXPORTS):
        return set()
    # No other read, alias, attribute or rebinding of `math` in the module.
    refs = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "math"]
    if refs != [module] or any(isinstance(node, ast.alias) and node.asname == "math" for node in ast.walk(tree)):
        return set()
    return {id(imports[0])}


def python_math_errors(path, relative):
    """Only the existing compensated-sum guard may call CPython math.fsum.

    No libm transcendental, alias escape or dynamic module import is admitted.
    The fast sum is held byte-for-byte to the exact integer oracle in tests.
    """
    if relative in NUMPY_ORACLES:
        return []  # independent test expectations, not runtime arithmetic
    tree = ast.parse(path.read_bytes(), filename=relative)
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    admitted = set()
    if relative == "mojolearn/_portable_math.py":
        sums = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'fsum']
        inside = {id(node) for fn in sums for node in ast.walk(fn)}
        refs = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == '_cmath']
        def is_sum_call(ref):
            attr = parents.get(id(ref))
            call = parents.get(id(attr))
            return (id(ref) in inside and isinstance(attr, ast.Attribute) and attr.attr == 'fsum'
                    and isinstance(call, ast.Call) and call.func is attr)
        if refs and all(is_sum_call(ref) for ref in refs):
            admitted = {id(node) for node in tree.body if isinstance(node, ast.Import)
                        and len(node.names) == 1 and node.names[0].name == 'math'
                        and node.names[0].asname == '_cmath'}
    if relative == "mojolearn/_native_loaded_modules.py":
        admitted = exact_math_connection_imports(tree)
    errors = []
    for node in ast.walk(tree):
        names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                 else [node.module or ''] if isinstance(node, ast.ImportFrom) and not node.level else [])
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            fn = node.func
            if (isinstance(fn, ast.Name) and fn.id == '__import__'
                    or isinstance(fn, ast.Attribute) and fn.attr == 'import_module'):
                names.append(str(node.args[0].value))
        if any(name.split('.')[0] in ('math', 'cmath') for name in names) and id(node) not in admitted:
            errors.append(relative + ': platform Python math import')
    return errors


def audit_tree(root, python_only=False):
    """Audit an unpacked wheel. `python_only` is the release rehearsal's
    dry run over a STAGED SOURCE TREE (tools/release_rehearsal.py): the same
    NumPy, dependency and Python-math rules over every shipped .py, with the
    native checks off, because a source tree's binaries have not been through
    stage() yet and would fail for that reason alone."""
    if not python_only:
        import lief
    errors, binaries = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        errors.extend(numpy_errors(path, relative))
        errors.extend(dependency_errors(path, relative))
        if LIBM.match(path.name):
            errors.append(relative + ": bundled platform math library")
        if path.suffix == ".py":
            errors.extend(python_math_errors(path, relative))
        if python_only:
            continue
        with path.open("rb") as stream:
            magic = stream.read(4)
        if magic == b"\x7fELF":
            binary = lief.ELF.parse(str(path))
            if binary is None:
                raise ValueError("unreadable ELF: " + relative)
            deps = list(binary.libraries)
            imports = [s.name.split("@")[0] for s in binary.imported_symbols]
        elif magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe"):
            fat = lief.MachO.parse(str(path))
            if fat is None or len(fat) != 1:
                raise ValueError("expected thin Mach-O: " + relative)
            binary = fat.at(0)
            deps = [x.name for x in binary.libraries]
            imports = [s.name.removeprefix("_") for s in binary.imported_symbols]
        else:
            continue
        if path.name in ("libMojolearnMath.so", "libMojolearnMath.dylib"):
            exported = {symbol.name.removeprefix("_") for symbol in binary.exported_symbols}
            required = {"mojolearn_" + name for name in
                        ("sqrt", "log", "log2", "log10", "log2f", "exp", "frexp", "ldexp", "modf", "lround")}
            if imports or not required <= exported:
                errors.append(relative + ": owned helper has unresolved imports or missing exports")
        bad_symbols = sorted(set(imports) & MATH_SYMBOLS)
        bad_deps = [dep for dep in deps if LIBM.match(Path(dep).name)]
        # FAST promises no bits across machines (per-vendor speed tier), so a
        # FAST-tier GPU set may call the platform's math; it is recorded, not
        # refused. IDENTICAL and host binaries stay enforced.
        fast_tier = bool(FAST_SET.match(relative))
        if (bad_symbols or bad_deps) and not fast_tier:
            errors.append(f"{relative}: math imports={bad_symbols}, dependencies={bad_deps}")
        binaries.append({"file": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "math_imports": bad_symbols, "math_dependencies": bad_deps,
                         **({"fast_tier_exempt": True} if fast_tier and (bad_symbols or bad_deps) else {})})
    if python_only:
        if errors:
            raise ValueError("platform math audit failed:\n" + "\n".join(errors))
        return {"numpy_optional": True, "numpy_bundled": False,
                "python_math_policy": "owned helpers and guarded CPython fsum", "binaries": [],
                "scope": "staged source tree, Python files only (release rehearsal)"}
    # IDENTICAL-tier CUDA PTX must carry a rounding modifier on every float
    # mul/add/sub/fma, or the driver JIT may contract it (packaging/linux/ptx_contract.py).
    ptx_errors, ptx_rows = ptx_contract.audit_tree(root)
    errors.extend(ptx_errors)
    # ...and must carry machine code for its set's architecture, so the
    # user's driver never JIT-compiles it (packaging/linux/cubin_contract.py).
    fatbin_errors, fatbin_rows = cubin_contract.audit_tree(root)
    errors.extend(fatbin_errors)
    if (root / "mojolearn/_portable_math.py").exists() and not any(
            (root / "mojolearn" / path).is_file() for path in
            (".libs/libMojolearnMath.so", ".dylibs/libMojolearnMath.dylib")):
        errors.append("owned Python math helper is missing its native library")
    baseline_reports = []
    baseline_root = root / "mojolearn/cuda_ptx"
    if baseline_root.exists():
        for directory in sorted(baseline_root.iterdir()):
            if not directory.is_dir() or directory.name != "sm_80":
                errors.append("unregistered portable PTX directory: " + str(directory))
                continue
            try:
                manifest = json.loads((directory / "PTX_BASELINE.json").read_text())
                report = ptx_baseline.audit_tree(directory, manifest["source_commit"],
                                                manifest["mojo_version"], manifest.get("source_dirty", False))
                errors.extend(report["errors"])
                # Independently recompute every byte/target/arithmetic field.
                # Retained component origins are additional provenance, not a
                # different binary format. Validate them against the recomputed
                # bytes before projecting just those known annotations away.
                import runpy
                plugins = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'python/mojolearn/gpu_plugins.py'))
                plugins['validate_baseline_manifest'](manifest,
                    {r['file']:r['sha256'] for r in report['files']},
                    {r['file']:r['sha256'] for r in report.get('indexes', [])})
                format_manifest = {k:v for k,v in manifest.items()
                                   if k not in ('component_sources', 'assembly_scope', 'base_source_commit',
                                                'release_admission')}
                admission = manifest.get('release_admission')
                if admission is not None:
                    # A RETAINED PTX LEG'S MANIFEST REBOUND FOR THIS RELEASE
                    # (packaging/linux/ptx_reuse.py; pack_wheel proved it against
                    # the set's bytes and both build proofs). Here: its raw
                    # original build manifest is intact and is this manifest
                    # with only the source commit moved, so every recomputed
                    # byte/target/arithmetic field above is the original build's.
                    raw = (admission.get('original_manifest') or {}).get('utf8', '')
                    original = json.loads(raw) if isinstance(raw, str) and raw else None
                    restored = dict(format_manifest, source_commit=(original or {}).get('source_commit'))
                    if (admission.get('schema') != 'mojolearn.ptx-release-admission.v1' or original is None
                            or hashlib.sha256(raw.encode()).hexdigest() != admission['original_manifest'].get('sha256')
                            or {k: v for k, v in original.items() if k not in ('component_sources', 'assembly_scope',
                                                                               'base_source_commit')} != restored):
                        errors.append("portable PTX release admission does not restore its original build manifest")
                if report != format_manifest:
                    errors.append("portable PTX final bytes differ from their build manifest")
                baseline_reports.append(report)
            except (OSError, ValueError, KeyError) as exc:
                errors.append("portable PTX manifest is missing or invalid: " + str(exc))
    aggregate = False
    aggregate_markers = list(root.glob("*.dist-info/gpu_plugin.json"))
    if len(aggregate_markers) == 1:
        try:
            aggregate = json.loads(aggregate_markers[0].read_text()).get("role") == "aggregate"
        except ValueError:
            pass
    if aggregate and any(p.is_file() and not p.relative_to(root).parts[0].endswith(".dist-info")
                         for p in root.rglob("*")):
        errors.append("vendor aggregate contains non-metadata files")
    if not binaries and not aggregate:
        errors.append("wheel contains no native binaries")
    if errors:
        raise ValueError("platform math audit failed:\n" + "\n".join(errors))
    return {"numpy_optional": True, "numpy_bundled": False, "numpy_oracles": sorted(NUMPY_ORACLES),
            "python_math_policy": "owned helpers and guarded CPython fsum", "platform_math_free": True,
            "scope": "wheel native math imports and Python operation policy; excludes Python and OS dependencies", "binaries": binaries,
            "identical_ptx_rounding_pinned": True, "identical_ptx": ptx_rows,
            # The native sets carry machine code; the PTX slot (cuda_ptx/sm_80) carries
            # rounding-pinned PTX and is reported separately (Andrew 2026-10-10:
            # PTX is a normal target; no flag).
            "identical_cuda_machine_code": True, "identical_cuda_fatbins": fatbin_rows,
            "ptx_sets": baseline_reports}


def finalize(wheel, helper=None, audit_only=False):
    wheel = Path(wheel)
    with tempfile.TemporaryDirectory(prefix="mojolearn-math-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            infos = {i.filename: i for i in archive.infolist()}
            for name in infos:
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("unsafe wheel member: " + name)
            archive.extractall(root)
        changes = []
        if not audit_only:
            runtime_dirs = sorted(p for p in (root / "mojolearn").rglob("*")
                                  if p.is_dir() and p.name in (".libs", ".dylibs"))
            if not runtime_dirs:
                raise ValueError("no bundled runtime directory")
            linux = any(p.name == ".libs" for p in runtime_dirs)
            import sys
            for directory in runtime_dirs:
                selected_helper = helper
                if linux and sys.platform != "linux" and selected_helper is None:
                    selected_helper = directory / "libMojolearnMath.so"
                    if not selected_helper.is_file():
                        raise ValueError("old Linux sets require --helper built on Linux; new build_sets outputs include it")
                changes.append({"directory": directory.relative_to(root).as_posix(), **stage(directory, selected_helper)})
            # Python host math loads one vendor-neutral helper by an exact path.
            if linux and not (root / "mojolearn/.libs/libMojolearnMath.so").exists():
                import shutil
                destination = root / "mojolearn/.libs/libMojolearnMath.so"
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(runtime_dirs[0] / "libMojolearnMath.so", destination)
        report = audit_tree(root)
        report["runtime_changes"] = changes
        if audit_only:
            return report
        dist = list(root.glob("*.dist-info"))
        if len(dist) != 1:
            raise ValueError("expected one dist-info")
        (dist[0] / "portable-math.json").write_text(json.dumps(report, indent=2) + "\n")
        record_path = dist[0] / "RECORD"
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path == record_path:
                continue
            data = path.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            writer.writerow([path.relative_to(root).as_posix(), "sha256=" + digest, len(data)])
        writer.writerow([record_path.relative_to(root).as_posix(), "", ""])
        record_path.write_text(output.getvalue())
        target = wheel.with_suffix(".whl.portable.tmp")
        try:
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        name = path.relative_to(root).as_posix()
                        archive.writestr(infos.get(name, name), path.read_bytes())
            target.replace(wheel)
        finally:
            target.unlink(missing_ok=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="*", type=Path)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--python-tree", type=Path, metavar="DIR",
                        help="audit the .py files of a staged package tree (the wheel's root layout) and exit")
    args = parser.parse_args()
    if args.python_tree:
        report = audit_tree(args.python_tree, python_only=True)
        print(json.dumps({"tree": str(args.python_tree), "numpy_optional": report["numpy_optional"],
                          "python_math_policy": report["python_math_policy"]}))
        return
    if not args.wheels:
        parser.error("name at least one wheel, or --python-tree DIR")
    for wheel in args.wheels:
        report = finalize(wheel, args.helper, args.audit_only)
        print(json.dumps({"wheel": str(wheel), "platform_math_free": True, "numpy_optional": report["numpy_optional"],
                          "native_files_checked": len(report["binaries"])}))


if __name__ == "__main__":
    main()
