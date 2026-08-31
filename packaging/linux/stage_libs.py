#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Stage the FULL transitive ELF closure of MAX runtime `.so` beside a set.

The Linux twin of `packaging/macos/stage_dylibs.py`, and it inherits that
file's lesson whole: a dependency graph has to be WALKED, not sampled. The
macOS stager read only the extension's direct dependencies once, staged two
dylibs, and shipped a wheel that failed on install because a staged dylib had
dependencies of its own. The real closure there was four. Here it is
unmeasured until the first build leg writes the manifest, and this script
refuses to guess: every `DT_NEEDED` of every staged file is resolved or the
run fails.

WHY THE BUILD BOX CANNOT SEE THE PROBLEM. On the box that compiled the
extension the RUNPATH still points into the pixi environment, so every
missing library is found anyway and the set imports perfectly. The failure
appears only on a machine without that environment, which is every user's.
So this does not rely on a runtime test. `verify_closed` proves STATICALLY
that every non-system `DT_NEEDED` of every staged file resolves inside
`.libs/`, which is checkable on the build box and is what the wheel needs.

THREE CLASSES OF DEPENDENCY, and the sort is the whole job:

  STAGED     found in the pixi environment's lib directory: the MAX runtime
             (`libAsyncRTMojoBindings.so`, `libKGENCompilerRTShared.so`,
             `libMSupportGlobals.so`, whatever else the walk finds). Copied
             into `.libs/`, given `$ORIGIN` as RUNPATH so they find each
             other. THEIR SIZE IS THE ONE NUMBER THIS WHOLE PLAN HAD TO
             MEASURE (docs/LINUX_WHEEL.md); the manifest records it.
  SYSTEM     glibc, libstdc++, libgcc_s, libm, libpthread, libdl, librt,
             ld-linux. Never staged; the manylinux policy decides whether
             the versions they are linked against are acceptable, and
             `auditwheel show` is the artifact that says so.
  DRIVER     the vendor's driver-side runtime: `libcuda.so.1`,
             `libnvidia-*`, `libamdhip64.so*`, `libhsa-runtime64.so*`,
             `libhsakmt*`, `libdrm*`. NEVER staged: they belong to the
             driver installed on the user's box, they are hundreds of
             megabytes, and bundling one pins a user to the driver the
             build box happened to have. Recorded in the manifest so the
             audit step can `--exclude` them by name rather than by guess.

Anything outside those three classes is UNRESOLVED and fails the run.

RUNPATH IS SET WITH BOTH CANDIDATE LOCATIONS so the packer on the Mac can
choose the layout without rewriting a single ELF header (there is no ELF
tool on the Mac and this plan runs nothing there beyond a pure-Python zip):

    <set root>/_mojolearn_x.so           $ORIGIN/.libs:$ORIGIN/../.libs:$ORIGIN/../../.libs
    <set root>/<tier>/_mojolearn_x.so    $ORIGIN/../.libs:$ORIGIN/../../.libs:$ORIGIN/../../../.libs
    .libs/lib*.so                        $ORIGIN

The set root is `<vendor>/<arch>/` in the wheel (the architecture axis,
2026-08-30), so the candidates cover a `.libs/` inside the set, one beside
the architecture directories (`mojolearn/<vendor>/.libs`), and one at the
package root (`mojolearn/.libs`, the shared layout when every closure is
byte-identical, which is what 2026-08-30 measured). A candidate that does
not exist is skipped by the loader at no cost. `pack_wheel.py` picks.

The dynamic section is parsed HERE, in pure Python, so the same reader can
inspect a fetched set on the Mac. Writing RUNPATH needs `patchelf`, which
`build_sets.sh` installs into a throwaway venv on the box.

Usage:
    stage_libs.py --set <vendor-set-dir> --env-lib <pixi env lib dir>
                  --manifest <out.json> [--patchelf <path>]

The set directory holds `*.so` (fast) and `deterministic/*.so`,
`identical/*.so`; `.libs/` is created inside it.
"""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys

DT_NULL, DT_NEEDED, DT_STRTAB, DT_RPATH, DT_RUNPATH = 0, 1, 5, 15, 29

SYSTEM_PREFIXES = (
    "libc.so", "libm.so", "libdl.so", "libpthread.so", "librt.so",
    "libstdc++.so", "libgcc_s.so", "ld-linux", "libutil.so", "libresolv.so",
)
DRIVER_PREFIXES = (
    "libcuda.so", "libnvidia-", "libcudart", "libnvrtc", "libnvptxcompiler",
    "libamdhip64.so", "libhsa-runtime64.so", "libhsakmt", "libdrm",
    "libhiprtc", "libamd_comgr", "librocm",
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def elf_dynamic(path):
    """(needed, runpath, rpath) from an ELF64 little-endian file's dynamic
    section. Raises on anything that is not one; a wheel must never carry a
    file this reader cannot classify."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF":
        raise SystemExit(f"stage_libs: {path} is not an ELF file")
    if data[4] != 2 or data[5] != 1:
        raise SystemExit(f"stage_libs: {path} is not ELF64 little-endian")
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        (sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link,
         sh_info, sh_addralign, sh_entsize) = struct.unpack_from(
            "<IIQQQQIIQQ", data, off)
        secs.append(dict(type=sh_type, addr=sh_addr, offset=sh_offset,
                         size=sh_size, link=sh_link, entsize=sh_entsize))
    dyn = [s for s in secs if s["type"] == 6]  # SHT_DYNAMIC
    if not dyn:
        return [], None, None
    dyn = dyn[0]
    strtab = secs[dyn["link"]]
    strs = data[strtab["offset"]:strtab["offset"] + strtab["size"]]

    def cstr(o):
        e = strs.index(b"\0", o)
        return strs[o:e].decode()

    needed, runpath, rpath = [], None, None
    n = dyn["size"] // 16
    for i in range(n):
        tag, val = struct.unpack_from("<qQ", data, dyn["offset"] + i * 16)
        if tag == DT_NULL:
            break
        if tag == DT_NEEDED:
            needed.append(cstr(val))
        elif tag == DT_RUNPATH:
            runpath = cstr(val)
        elif tag == DT_RPATH:
            rpath = cstr(val)
    return needed, runpath, rpath


def classify(name, env_lib):
    if any(name.startswith(p) for p in SYSTEM_PREFIXES):
        return "system"
    if any(name.startswith(p) for p in DRIVER_PREFIXES):
        return "driver"
    if (env_lib / name).exists():
        return "staged"
    return "unresolved"


def closure(exts, env_lib):
    """Every STAGED library reachable from `exts`, in discovery order, plus
    the full classified edge list for the manifest."""
    order, seen, edges, driver, unresolved = [], set(), [], set(), []
    queue = list(exts)
    while queue:
        cur = queue.pop(0)
        needed, _, _ = elf_dynamic(cur)
        for dep in needed:
            cls = classify(dep, env_lib)
            edges.append({"from": os.path.basename(str(cur)), "needs": dep,
                          "class": cls})
            if cls == "driver":
                driver.add(dep)
            elif cls == "unresolved":
                unresolved.append(f"{os.path.basename(str(cur))} -> {dep}")
            elif cls == "staged" and dep not in seen:
                seen.add(dep)
                order.append(dep)
                queue.append(env_lib / dep)
    if unresolved:
        print("ERROR: dependencies that are neither system, driver nor in "
              f"{env_lib}:", file=sys.stderr)
        for u in unresolved:
            print("   ", u, file=sys.stderr)
        raise SystemExit(1)
    return order, edges, sorted(driver)


def set_runpath(patchelf, path, value):
    subprocess.run([patchelf, "--set-rpath", value, str(path)], check=True)


def verify_closed(files, libs_dir, env_lib):
    have = {p.name for p in libs_dir.glob("*.so*")}
    missing = []
    for f in files:
        needed, _, _ = elf_dynamic(f)
        for dep in needed:
            if classify(dep, env_lib) == "staged" and dep not in have:
                missing.append(f"{f.name} -> {dep}")
    if missing:
        print("ERROR: the staged library set is NOT closed:", file=sys.stderr)
        for m in missing:
            print("   ", m, file=sys.stderr)
        raise SystemExit(1)
    print(f"  closure verified: {len(have)} libraries, nothing unresolved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--env-lib", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--patchelf", default=shutil.which("patchelf"))
    a = ap.parse_args()
    root = pathlib.Path(a.set).resolve()
    env_lib = pathlib.Path(a.env_lib).resolve()
    if not a.patchelf:
        raise SystemExit("stage_libs: patchelf not found; build_sets.sh installs one")

    exts = sorted(root.glob("_mojolearn*.so"))
    for tier in ("deterministic", "identical"):
        exts += sorted((root / tier).glob("_mojolearn*.so"))
    if not exts:
        raise SystemExit(f"stage_libs: no _mojolearn*.so under {root}")

    libs = root / ".libs"
    if libs.exists():
        shutil.rmtree(libs)
    libs.mkdir()
    needed, edges, driver = closure(exts, env_lib)
    staged = []
    for name in needed:
        src = env_lib / name
        # Follow symlinks: the pixi env may expose libfoo.so -> libfoo.so.1.
        shutil.copy2(src.resolve(), libs / name)
        set_runpath(a.patchelf, libs / name, "$ORIGIN")
        staged.append({"name": name, "bytes": (libs / name).stat().st_size,
                       "sha256": sha256(libs / name),
                       "source": str(src.resolve())})
    ext_records = []
    for ext in exts:
        if ext.parent == root:
            rp = "$ORIGIN/.libs:$ORIGIN/../.libs:$ORIGIN/../../.libs"
            tier = "fast"
        else:
            rp = "$ORIGIN/../.libs:$ORIGIN/../../.libs:$ORIGIN/../../../.libs"
            tier = ext.parent.name
        set_runpath(a.patchelf, ext, rp)
        n, runpath, rpath = elf_dynamic(ext)
        ext_records.append({"path": str(ext.relative_to(root)), "tier": tier,
                            "bytes": ext.stat().st_size, "sha256": sha256(ext),
                            "needed": n, "runpath": runpath})
    verify_closed(exts + sorted(libs.glob("*.so*")), libs, env_lib)

    total_ext = sum(r["bytes"] for r in ext_records)
    total_libs = sum(r["bytes"] for r in staged)
    manifest = {
        "set": str(root), "env_lib": str(env_lib),
        "extensions": ext_records, "staged_libs": staged,
        "driver_libs_not_staged": driver, "edges": edges,
        "bytes_extensions": total_ext, "bytes_staged_libs": total_libs,
        "bytes_total_uncompressed": total_ext + total_libs,
    }
    with open(a.manifest, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"  staged {len(staged)}: {', '.join(r['name'] for r in staged)}")
    print(f"  driver libraries left to the box: {driver or 'none'}")
    print(f"  extensions {total_ext/1e6:.1f} MB, runtime libraries "
          f"{total_libs/1e6:.1f} MB, uncompressed set {(total_ext+total_libs)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
