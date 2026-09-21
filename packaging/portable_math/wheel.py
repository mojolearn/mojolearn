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

# C99 math entry points, including float/long-double variants. A new runtime
# dependency must fail closed instead of being silently removed by the patcher.
ROOTS = "acos acosh asin asinh atan atan2 atanh cbrt ceil copysign cos cosh erf erfc exp exp2 expm1 fabs fdim floor fma fmax fmin fmod frexp hypot ilogb ldexp lgamma llrint llround log log10 log1p log2 logb lrint lround modf nearbyint nextafter nexttoward pow remainder remquo rint round scalbln scalbn sin sincos sinh sqrt tan tanh tgamma trunc".split()
MATH_SYMBOLS = {root + suffix for root in ROOTS for suffix in ("", "f", "l")}
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
               for dep in metadata.get_all("Requires-Dist", [])):
            errors.append(relative + ": NumPy dependency metadata")
    if path.suffix == ".py" and relative not in NUMPY_ORACLES:
        for node in ast.walk(ast.parse(path.read_bytes(), filename=relative)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                func = node.func
                if (isinstance(func, ast.Name) and func.id == "__import__"
                        or isinstance(func, ast.Attribute) and func.attr == "import_module"):
                    names.append(str(node.args[0].value))
            if any(name.split(".")[0] == "numpy" for name in names):
                errors.append(relative + ": NumPy import outside independent verification")
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


def dependency_errors(path, relative):
    """A shipped .py importing anything but the standard library and mojolearn
    (NumPy is judged by numpy_errors; an optional interop package only lazily)."""
    if path.suffix != ".py":
        return []
    import sys
    allowed = set(sys.stdlib_module_names) | {"mojolearn", "numpy", "__future__"}
    errors = []
    tree = ast.parse(path.read_bytes(), filename=relative)
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
            if top in SOURCE_TREE_LAZY_IMPORTS.get(relative, ()) and id(node) in lazy:
                continue
            errors.append(f"{relative}: imports {top!r}, which the wheel does not provide")
    return errors


LIBM = re.compile(r"^lib(?:m|mvec)(?:[.-]|$)", re.I)


def audit_tree(root):
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
            tree = ast.parse(path.read_bytes(), filename=relative)
            for node in ast.walk(tree):
                imports = ([a.name for a in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
                if any(name.split(".")[0] in ("math", "cmath") for name in imports):
                    errors.append(relative + ": platform Python math import")
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
        if bad_symbols or bad_deps:
            errors.append(f"{relative}: math imports={bad_symbols}, dependencies={bad_deps}")
        binaries.append({"file": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "math_imports": bad_symbols, "math_dependencies": bad_deps})
    if (root / "mojolearn/_portable_math.py").exists() and not any(
            (root / "mojolearn" / path).is_file() for path in
            (".libs/libMojolearnMath.so", ".dylibs/libMojolearnMath.dylib")):
        errors.append("owned Python math helper is missing its native library")
    if not binaries:
        errors.append("wheel contains no native binaries")
    if errors:
        raise ValueError("platform math audit failed:\n" + "\n".join(errors))
    return {"numpy_runtime_free": True, "numpy_oracles": sorted(NUMPY_ORACLES), "platform_math_free": True, "scope": "wheel files and direct native/Python math imports; excludes Python and OS dependencies", "binaries": binaries}


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
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    for wheel in args.wheels:
        report = finalize(wheel, args.helper, args.audit_only)
        print(json.dumps({"wheel": str(wheel), "platform_math_free": True, "numpy_runtime_free": report["numpy_runtime_free"],
                          "native_files_checked": len(report["binaries"])}))


if __name__ == "__main__":
    main()
