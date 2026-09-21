#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build owned host math and redirect bundled runtime math imports to it.

Requires LIEF 1.0+ in the build environment. Existing runtime instruction
bodies are not rebuilt. Import names are namespaced, so a Python process that
already loaded platform libm cannot accidentally satisfy these imports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

NAMES = {name: "mojolearn_" + name for name in ("log", "log2", "log2f", "log10", "frexp", "ldexp", "lround", "modf")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, cc=None):
    source = Path(__file__).with_name("portable_math.c")
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler = cc or os.environ.get("CC", "clang" if sys.platform == "darwin" else "cc")
    args = [compiler, "-O2", "-ffp-contract=off", "-fno-fast-math"]
    if sys.platform == "darwin":
        # Apple's dynamic linker requires libSystem even with no undefined
        # symbols. No math symbol is imported from that OS ABI library.
        args += ["-dynamiclib", "-arch", "arm64", "-mmacosx-version-min=11.0",
                 "-Wl,-install_name,@rpath/libMojolearnMath.dylib"]
    else:
        # x86-64-v3 is the Linux wheel's baseline. The CPU identity gate also
        # builds this helper on ARM64 Linux, where that flag is an error; the
        # source's aarch64 branch uses only base AArch64 instructions (fmadd,
        # fsqrt), so armv8-a is its baseline there.
        import platform
        machine = platform.machine().lower()
        march = "-march=armv8-a" if machine in ("aarch64", "arm64") else "-march=x86-64-v3"
        args += ["-shared", "-fPIC", march, "-nostdlib",
                 "-Wl,-soname,libMojolearnMath.so"]
    subprocess.run(args + [str(source), "-o", str(output)], check=True)
    if sys.platform == "darwin":
        subprocess.run(["codesign", "--force", "--sign", "-", str(output)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"source_sha256": sha(source),
            "constants_sha256": sha(source.with_name("powers_of_ten.h")),
            "output_sha256": sha(output), "command": args}


def patch(path):
    import lief
    before = sha(path)
    rewrites = []
    removed = []
    prefix = path.read_bytes()[:4]
    if prefix == b"\x7fELF":
        binary = lief.ELF.parse(str(path))
        for symbol in binary.imported_symbols:
            if symbol.name in NAMES:
                old = symbol.name
                symbol.name = NAMES[old]
                if symbol.symbol_version is not None:
                    symbol.symbol_version.as_global()
                rewrites.append([old, symbol.name])
        for name in list(binary.libraries):
            if name.startswith("libm.so"):
                binary.remove_version_requirement(name)
                binary.remove_library(name)
                removed.append(name)
        if rewrites and "libMojolearnMath.so" not in binary.libraries:
            binary.add_library("libMojolearnMath.so")
        if rewrites or removed:
            temporary = path.with_name(path.name + ".portable.tmp")
            binary.write(str(temporary))
            temporary.replace(path)
    else:
        fat = lief.MachO.parse(str(path))
        if fat is None or len(fat) != 1:
            raise ValueError("expected a single-architecture wheel Mach-O: " + str(path))
        binary = fat.at(0)
        symbols = [s for s in binary.imported_symbols if s.name.startswith("_") and s.name[1:] in NAMES]
        if symbols:
            helper = "@rpath/libMojolearnMath.dylib"
            loads = [x for x in binary.libraries if x.command != lief.MachO.LoadCommand.TYPE.ID_DYLIB]
            if not any(x.name == helper for x in loads):
                binary.add_library(helper)
            loads = [x for x in binary.libraries if x.command != lief.MachO.LoadCommand.TYPE.ID_DYLIB]
            ordinal = next(i + 1 for i, x in enumerate(loads) if x.name == helper)
            for symbol in symbols:
                old = symbol.name
                symbol.name = "_" + NAMES[old[1:]]
                symbol.description = (symbol.description & 255) | (ordinal << 8)
                rewrites.append([old, symbol.name])
            renamed = {new for _, new in rewrites}
            for binding in binary.bindings:
                if binding.has_symbol and binding.symbol.name in renamed:
                    binding.library_ordinal = ordinal
            temporary = path.with_name(path.name + ".portable.tmp")
            binary.write(str(temporary))
            temporary.replace(path)
            if sys.platform == "darwin":
                subprocess.run(["codesign", "--force", "--sign", "-", str(path)], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"file": path.name, "before_sha256": before, "after_sha256": sha(path),
            "rewritten_imports": rewrites, "removed_dependencies": removed}


def stage(directory, helper=None, cc=None):
    directory = Path(directory)
    name = "libMojolearnMath.dylib" if any(directory.glob("*.dylib")) else "libMojolearnMath.so"
    destination = directory / name
    if helper is None:
        build_receipt = build(destination, cc)
    else:
        import shutil
        if Path(helper).resolve() != destination.resolve():
            shutil.copy2(helper, destination)
        build_receipt = {"supplied_helper_sha256": sha(destination)}
    records = [patch(p) for p in sorted(directory.iterdir())
               if p.is_file() and p != destination and p.suffix in (".so", ".dylib")]
    return {"helper": name, "build": build_receipt, "runtime_changes": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--cc")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report = stage(args.directory, args.helper, args.cc)
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
