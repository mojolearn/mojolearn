#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Stage the FULL transitive closure of @rpath dylibs beside the extension.

WHY A TRANSITIVE WALK AND NOT `otool -L` ONCE. The first version of this
staging read only the extension's DIRECT dependencies, found two dylibs,
copied them, and declared victory. It shipped a wheel that failed on install
with

    Library not loaded: @rpath/libMSupportGlobals.dylib
      Referenced from: .../.dylibs/libAsyncRTMojoBindings.dylib

because a dylib we staged had dependencies of its own. The real closure is
four, not two. A dependency graph has to be walked, not sampled.

WHY THE BUILD MACHINE COULD NOT SEE IT. On the machine that compiled the
extension the ORIGINAL @rpath still resolves into the pixi environment, so
every missing dylib is found anyway and the wheel imports perfectly. The
failure appears only on a machine without that environment. This is the same
trap the wheel-verification note already warned about, one level deeper: it is
not enough to test in a clean venv, because a clean venv on the build machine
still has the build machine's filesystem.

So this script does not rely on a runtime test at all. `verify_closed` proves
STATICALLY that every @rpath dependency of every staged file resolves inside
.dylibs/, which is checkable on the build machine and is what the wheel
actually needs to be true.

Each staged dylib also gets @loader_path added to its OWN rpath, because they
reference each other, not just the extension.

Everything is re-signed: macOS invalidates a signature the moment
install_name_tool rewrites a load command, and an unsigned dylib is killed on
load on Apple silicon rather than merely warned about.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from native_library_sources import library_source, file_sha

# zlib, bzip2 and liblzma use macOS /usr/lib ABIs. Zstd is not a system
# library: native archive decoding opens @rpath/libzstd.1.dylib explicitly.
ARCHIVE_CODEC_LIBRARIES = ("libzstd.1.dylib",)


def workers(units):
    return max(1, min(units, os.cpu_count() or 1))



def rpath_deps(path):
    """The @rpath-relative dylib names this Mach-O file requires."""
    out = subprocess.run(
        ["otool", "-L", str(path)], capture_output=True, text=True
    )
    if out.returncode != 0:
        raise SystemExit(f"otool failed on {path}")
    deps = []
    for line in out.stdout.splitlines()[1:]:
        line = line.strip()
        if line.startswith("@rpath/"):
            deps.append(line.split()[0][len("@rpath/") :])
    return deps


def closure(root, env_lib, dynamic=(), codec_lib=None, sources=None):
    """Declared dlopen roots and @rpath dependencies, by parallel frontiers."""
    codec_lib = env_lib if codec_lib is None else codec_lib
    selected = {} if sources is None else sources
    queue = list(root) if isinstance(root, (list, tuple)) else [root]
    seen, order = set(), []
    for dep in dynamic:
        source = library_source(dep, env_lib, env_lib, codec_lib, selected, "native archive codec")
        if dep not in seen:
            seen.add(dep)
            order.append(dep)
            queue.append(source)
    while queue:
        frontier, queue = queue, []
        with ThreadPoolExecutor(max_workers=workers(len(frontier))) as pool:
            results = list(pool.map(rpath_deps, frontier))
        for cur, dependencies in zip(frontier, results):
            for dep in dependencies:
                preferred = codec_lib if pathlib.Path(cur).parent == codec_lib else env_lib
                candidate = library_source(dep, preferred, env_lib, codec_lib, selected)
                if dep in seen:
                    continue
                seen.add(dep)
                order.append(dep)
                queue.append(candidate)
    return order


def absolute_rpaths(path):
    """LC_RPATH entries that are absolute filesystem paths."""
    out = subprocess.run(
        ["otool", "-l", str(path)], capture_output=True, text=True
    ).stdout
    found, lines = [], out.splitlines()
    for i, line in enumerate(lines):
        if "LC_RPATH" in line:
            for j in range(i, min(i + 5, len(lines))):
                t = lines[j].strip()
                if t.startswith("path ") and t.split()[1].startswith("/"):
                    found.append(t.split()[1])
                    break
    return found


def strip_build_rpaths(path):
    """Remove absolute rpaths pointing at the machine that built this.

    THREE REASONS, and the first is the one that is easy to miss.

    1. It is why a local test cannot fail. The extension carries
       `/Users/<someone>/.../.pixi/envs/default/lib`, so on the build machine
       every dylib resolves through it whether or not staging worked. Removing
       it makes `verify_wheel.sh` a real test instead of a formality.
    2. It publishes a developer's home directory inside a binary on PyPI.
    3. It is a load path on a stranger's machine. If that exact path exists
       there, dyld will search it.
    """
    removed = []
    for rp in absolute_rpaths(path):
        r = subprocess.run(
            ["install_name_tool", "-delete_rpath", rp, str(path)],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            removed.append(rp)
    return removed


def sign(path):
    subprocess.run(
        ["codesign", "--force", "--sign", "-", str(path)],
        check=True, capture_output=True,
    )


def add_rpath(path, value):
    # Already present is fine; install_name_tool says so on stderr and exits
    # non-zero, which is not an error for our purposes.
    subprocess.run(
        ["install_name_tool", "-add_rpath", value, str(path)],
        capture_output=True,
    )


def verify_closed(ext, dylibs_dir, dynamic=()):
    """Every @rpath dep of every staged file must resolve inside .dylibs/.

    THIS IS THE GATE. It is static, so it holds on the build machine, where a
    runtime import proves nothing because the build environment is still on
    disk.
    """
    have = {p.name for p in dylibs_dir.glob("*.dylib")}
    missing = ["native archive dlopen -> " + name for name in dynamic if name not in have]
    for f in [ext, *sorted(dylibs_dir.glob("*.dylib"))]:
        for dep in rpath_deps(f):
            if dep not in have:
                missing.append(f"{f.name} -> {dep}")
    if missing:
        print("ERROR: the staged dylib set is NOT closed:", file=sys.stderr)
        for m in missing:
            print("   ", m, file=sys.stderr)
        raise SystemExit(1)
    print(f"  closure verified: {len(have)} dylibs, no unresolved @rpath")


def main():
    # ONE `.dylibs` SERVES EVERY EXTENSION IN THE PACKAGE, so every extension
    # has to be staged in ONE call.
    #
    # This took a list of exactly one until 2026-08-21, when GBDT moved into a
    # second extension. Calling it twice does not work and fails in the worst
    # available way: the first line of `main` WIPES `.dylibs`, so the second
    # call would delete the first extension's staged closure and verify only
    # its own. On the build machine the deleted dylibs are still resolvable
    # through the pixi environment, so the wheel would import perfectly here
    # and die on a user's Mac -- which is the exact trap this file's docstring
    # was written about, one extension wider.
    #
    # So: the closure is the UNION over all of them, and `verify_closed` runs
    # for EACH, because a union that satisfies one extension and not another
    # is still a broken wheel.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-lib", type=pathlib.Path,
                        help="pinned packaging environment library directory")
    parser.add_argument("paths", nargs="+", help="extensions followed by compiler environment lib")
    args = parser.parse_args()
    if len(args.paths) < 2:
        parser.error("provide at least one extension and the compiler environment lib directory")
    exts = [pathlib.Path(path) for path in args.paths[:-1]]
    env_lib = pathlib.Path(args.paths[-1]).resolve()
    codec_lib = args.codec_lib.resolve() if args.codec_lib else env_lib
    # The package directory is the FIRST extension's directory. Every other
    # extension sits there or exactly one directory below it (the
    # `identical/` numeric-mode set, 2026-08-23); anything else is refused,
    # because one `.dylibs` is staged and the rpath written into each
    # extension is computed from its depth.
    # The package directory is the one that holds __init__.py. The first extension used to sit
    # there; since the merged layout it sits one level below (identical/_mojolearn_native.so),
    # and taking its parent as the root refused the host/ member (0.8.56, 2026-10-24).
    root = exts[0].parent
    if not (root / "__init__.py").exists() and (root.parent / "__init__.py").exists():
        root = root.parent
    dylibs = root / ".dylibs"
    for ext in exts:
        if not ext.exists():
            raise SystemExit(f"stage_dylibs: no such extension: {ext}")
        if ext.parent != root and ext.parent.parent != root:
            raise SystemExit(
                "stage_dylibs: every extension must sit in the package "
                f"directory or one level below it; {ext} is not under {root}"
            )

    if dylibs.exists():
        shutil.rmtree(dylibs)
    dylibs.mkdir(parents=True)

    library_sources = {}
    needed = closure(exts, env_lib, ARCHIVE_CODEC_LIBRARIES, codec_lib, library_sources)
    def stage_library(name):
        shutil.copy2(library_sources[name], dylibs / name)
        add_rpath(dylibs / name, "@loader_path")
        # Signing happens after all load-command edits below.
    with ThreadPoolExecutor(max_workers=workers(len(needed))) as pool:
        list(pool.map(stage_library, needed))

    # REUSED EXTENSIONS (MOJOLEARN_STAGE_SKIP, build_release_wheel.sh,
    # 2026-09-23): taken from the published wheel, so already staged and
    # signed; rewriting a load command or re-signing would move bytes that
    # must stay the published ones. They keep their place in the closure and
    # in verify_closed.
    skip = {pathlib.Path(p).resolve() for p in os.environ.get("MOJOLEARN_STAGE_SKIP", "").split() if p}
    reused = [ext for ext in exts if ext.resolve() in skip]
    fresh = [ext for ext in exts if ext.resolve() not in skip]
    dropped = []
    for ext in fresh:
        up = "" if ext.parent == root else "../"
        add_rpath(ext, f"@loader_path/{up}.dylibs")
        dropped += strip_build_rpaths(ext)
    for name in needed:
        dropped += strip_build_rpaths(dylibs / name)
    for ext in fresh:
        sign(ext)
    for name in needed:
        sign(dylibs / name)
    if reused:
        print(f"  left untouched (reused, already staged and signed): "
              f"{', '.join(str(e.relative_to(root)) for e in reused)}")
    if dropped:
        print(f"  removed {len(dropped)} build-machine rpath(s):")
        for d in dropped:
            print("   ", d)

    for name in ARCHIVE_CODEC_LIBRARIES:
        print(f"  native archive input: {name} sha256={file_sha(library_sources[name])} source={library_sources[name].resolve()}")
    print(f"  staged {len(needed)}: {', '.join(needed)}")
    for ext in exts:
        print(f"  for {ext.relative_to(root)}:")
        verify_closed(ext, dylibs, ARCHIVE_CODEC_LIBRARIES)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: stage_dylibs.py <extension.so> [<extension.so> ...]"
            " <env_lib_dir>"
        )
        raise SystemExit(2)
    main()
