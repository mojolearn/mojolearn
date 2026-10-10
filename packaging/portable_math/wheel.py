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
#: with no tier directory (identical/ and deterministic/ sit one level deeper).
FAST_SET = re.compile(r"^mojolearn/(cuda|hip|cuda_native|hip_native|cuda_ptx)/[^/]+/_mojolearn[^/]*\.so$"
                      # macOS: FAST GPU bindings sit at the package root; host/,
                      # identical/ and deterministic/ are directories below it.
                      r"|^mojolearn/_mojolearn[^/]*\.so$")
# These entry points run independent tests; none is imported by estimators.
NUMPY_ORACLES = {
    "mojolearn/_identity_break.py", "mojolearn/_identity.py",
    "mojolearn/_verify_all.py", "mojolearn/_verify_distributed.py",
    "mojolearn/_verify_par.py",
    "mojolearn/_verify_parallel_cv.py", "mojolearn/_parallel_cv_witness.py",
}


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
        optional_imports = set()
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
            if top in OPTIONAL_LAZY_IMPORTS and id(node) in lazy:
                continue
            if id(node) in sparse_adapters:
                continue
            if top in SOURCE_TREE_LAZY_IMPORTS.get(relative, ()) and id(node) in lazy:
                continue
            errors.append(f"{relative}: imports {top!r}, which the wheel does not provide")
    return errors


LIBM = re.compile(r"^lib(?:m|mvec)(?:[.-]|$)", re.I)


def python_math_errors(path, relative):
    """Only the existing compensated-sum guard may call CPython math.fsum.

    No libm transcendental, alias escape or dynamic module import is admitted.
    The fast sum is held byte-for-byte to the exact integer oracle in tests.
    """
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
        # refused. IDENTICAL, deterministic and host binaries stay enforced.
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
                if report != manifest:
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
