#!/usr/bin/env python3
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

import pathlib
import shutil
import subprocess
import sys


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


def closure(root, env_lib):
    """Every dylib reachable from `root` through @rpath, breadth first."""
    seen, queue, order = set(), [root], []
    while queue:
        cur = queue.pop(0)
        for dep in rpath_deps(cur):
            if dep in seen:
                continue
            seen.add(dep)
            order.append(dep)
            cand = env_lib / dep
            if not cand.exists():
                raise SystemExit(
                    f"ERROR: {dep} is required by {pathlib.Path(cur).name} "
                    f"but is not in {env_lib}"
                )
            queue.append(cand)
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


def verify_closed(ext, dylibs_dir):
    """Every @rpath dep of every staged file must resolve inside .dylibs/.

    THIS IS THE GATE. It is static, so it holds on the build machine, where a
    runtime import proves nothing because the build environment is still on
    disk.
    """
    have = {p.name for p in dylibs_dir.glob("*.dylib")}
    missing = []
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
    exts = [pathlib.Path(a) for a in sys.argv[1:-1]]
    env_lib = pathlib.Path(sys.argv[-1])
    # The package directory is the FIRST extension's directory. Every other
    # extension sits there or exactly one directory below it (the
    # `identical/` numeric-mode set, 2026-08-23); anything else is refused,
    # because one `.dylibs` is staged and the rpath written into each
    # extension is computed from its depth.
    root = exts[0].parent
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

    needed = []
    for ext in exts:
        for name in closure(ext, env_lib):
            if name not in needed:
                needed.append(name)
    for name in needed:
        shutil.copy2(env_lib / name, dylibs / name)
        # They reference EACH OTHER, so each needs to find its siblings.
        add_rpath(dylibs / name, "@loader_path")
        sign(dylibs / name)

    dropped = []
    for ext in exts:
        up = "" if ext.parent == root else "../"
        add_rpath(ext, f"@loader_path/{up}.dylibs")
        dropped += strip_build_rpaths(ext)
    for name in needed:
        dropped += strip_build_rpaths(dylibs / name)
    for ext in exts:
        sign(ext)
    for name in needed:
        sign(dylibs / name)
    if dropped:
        print(f"  removed {len(dropped)} build-machine rpath(s):")
        for d in dropped:
            print("   ", d)

    print(f"  staged {len(needed)}: {', '.join(needed)}")
    for ext in exts:
        print(f"  for {ext.relative_to(root)}:")
        verify_closed(ext, dylibs)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: stage_dylibs.py <extension.so> [<extension.so> ...]"
            " <env_lib_dir>"
        )
        raise SystemExit(2)
    main()
