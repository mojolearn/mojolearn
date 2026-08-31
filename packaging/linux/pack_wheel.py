#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Pack the fetched CUDA and HIP sets into ONE Linux wheel. Pure Python.

    python3 packaging/linux/pack_wheel.py \\
        --set bench/results/wheels/<stamp1>-nvidia/sets/cuda \\
        --set bench/results/wheels/<stamp2>-nvidia/sets/cuda \\
        --set bench/results/wheels/<stamp3>-amd/sets/hip \\
        --out python/dist

EVERY SET CARRIES AN ARCHITECTURE LEVEL (2026-08-30): a `--set` directory is
`sets/<vendor>/` holding one or more `<arch>/` subdirectories (`sm_80`,
`gfx942`, ...), because one `mojo build` emits device code for exactly one
architecture and no PTX -- the sm_90a-only 0.3.0 wheel failed 27 of 29 lanes
on an A40 (LEGS_2026-08-30.md). Different architectures of one vendor come
from different legs, so the same vendor may be given several times; the same
(vendor, architecture) twice is refused. An arch-less set (binaries directly
under `sets/<vendor>/`) predates the axis and is REFUSED: rebuild it with
the current build_sets.sh rather than shipping a wheel that only runs on the
GPU model that built it.

RUNS ON THE MAC AND RUNS NOTHING. No setuptools, no compiler, no ELF tool:
it reads `python/pyproject.toml` (tomllib), `python/mojolearn/_version.py`,
the tracked `.py` files, the two set directories, and writes a zip with a
correct RECORD. It refuses if the two version files disagree, if a set is
missing a tier or a binding, if a set's `readback.txt` names a vendor other
than its directory, or if the finished archive is over PyPI's limit.

THE TAG IT WRITES IS `linux_x86_64`, DELIBERATELY. PyPI refuses that tag,
so the wheel this produces CANNOT be uploaded until `auditwheel` has looked
at it and rewritten the tag to the manylinux level it actually measured
(`packaging/linux/audit.sh`). The manylinux floor of the MAX runtime is one
of the numbers this plan has never measured, and a tag typed here would be
a guess wearing a measurement's clothes.

THE `.libs` LAYOUT IS DECIDED HERE, from the manifests. Every extension was
given BOTH candidate RUNPATHs on the box (`stage_libs.py`), so:

    both closures byte-identical   ONE mojolearn/.libs/ shared by cuda/ and hip/
    otherwise                      mojolearn/cuda/.libs/ and mojolearn/hip/.libs/

and either way no header is rewritten on the Mac. The choice, and the bytes
it saved, are printed and written into SIZES.json beside the wheel.

WHAT IS IN THE WHEEL, and why it matches the macOS one file for file except
for the binaries: `mojolearn_diagnostics.py`, every `.py` directly under
`python/mojolearn/` (no subpackage, no tests, no reference cards -- the
0.1.0 macOS wheel's listing is the reference, and it carries none of
those), `mojolearn-<v>.dist-info/{METADATA,WHEEL,RECORD,entry_points.txt,
top_level.txt,licenses/LICENSE,licenses/NOTICE}`. METADATA is generated
from pyproject.toml with the same field order setuptools 84 wrote for
0.1.0, and `--check-against <macos wheel>` diffs the two METADATA bodies so
a drift is a visible line rather than a silent difference between the two
artifacts of one release.
"""

import argparse
import base64
import hashlib
import json
import os
import pathlib
import re
import sys
import zipfile

try:
    import tomllib
except ImportError:  # pragma: no cover
    raise SystemExit("pack_wheel.py needs Python 3.11+ (tomllib)")

REPO = pathlib.Path(__file__).resolve().parents[2]
PY_DIR = REPO / "python"
PKG = PY_DIR / "mojolearn"
EXT_NAMES = (
    "_mojolearn", "_mojolearn_gbdt", "_mojolearn_estimators", "_mojolearn_rf",
    "_mojolearn_trees", "_mojolearn_svm", "_mojolearn_solver",
    "_mojolearn_metrics", "_mojolearn_tsa", "_mojolearn_linalg",
)
TIERS = ("fast", "deterministic", "identical")
ARCH_RE = re.compile(r"^(sm_[0-9]+a?|gfx[0-9a-f]+)$")
PYPI_LIMIT = 100 * 1024 * 1024
LINUX_VENDORS = ("cuda", "hip")


def urlsafe_b64(digest):
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def read_version():
    src = (PKG / "_version.py").read_text()
    m = re.search(r'__version__\s*=\s*"([^"]+)"', src)
    if not m:
        raise SystemExit("pack_wheel: no __version__ in _version.py")
    return m.group(1)


def metadata_text(proj, readme):
    """Metadata 2.4, field order as setuptools 84 wrote it for 0.1.0."""
    lines = ["Metadata-Version: 2.4", f"Name: {proj['name']}",
             f"Version: {proj['version']}", f"Summary: {proj['description']}"]
    if proj.get("authors"):
        lines.append("Author: " + ", ".join(a["name"] for a in proj["authors"]))
    if proj.get("maintainers"):
        lines.append("Maintainer: " + ", ".join(a["name"] for a in proj["maintainers"]))
    if proj.get("license"):
        lines.append(f"License-Expression: {proj['license']}")
    for k, v in proj.get("urls", {}).items():
        lines.append(f"Project-URL: {k}, {v}")
    if proj.get("keywords"):
        lines.append("Keywords: " + ",".join(proj["keywords"]))
    for c in proj.get("classifiers", []):
        lines.append(f"Classifier: {c}")
    if proj.get("requires-python"):
        lines.append(f"Requires-Python: {proj['requires-python']}")
    lines.append("Description-Content-Type: text/markdown")
    for lf in proj.get("license-files", []):
        lines.append(f"License-File: {lf}")
    for d in proj.get("dependencies", []):
        lines.append(f"Requires-Dist: {d}")
    lines.append("Dynamic: license-file")
    return "\n".join(lines) + "\n\n" + readme


def load_set(path):
    """Every (vendor, arch, files, libs, manifest) under one sets/<vendor>
    directory. One tuple per architecture subdirectory."""
    path = pathlib.Path(path).resolve()
    vendor = path.name
    if vendor not in LINUX_VENDORS:
        raise SystemExit(f"pack_wheel: set directory must be named cuda or hip: {path}")
    if any(path.glob("_mojolearn*.so")):
        raise SystemExit(
            f"pack_wheel: {path} holds binaries with NO architecture level. "
            "That set predates the architecture axis (2026-08-30) and would "
            "only run on the GPU model that built it -- the sm_90a/A40 "
            "failure. Rebuild it with packaging/linux/build_sets.sh.")
    arch_dirs = sorted(d for d in path.iterdir()
                       if d.is_dir() and ARCH_RE.match(d.name))
    if not arch_dirs:
        raise SystemExit(f"pack_wheel: no <arch>/ subdirectory under {path}")
    out = []
    for adir in arch_dirs:
        arch = adir.name
        manifest = json.loads((adir / "manifest.json").read_text())
        rb = (adir / "readback.txt").read_text().split()
        said = {w for w in rb if w in ("cuda", "hip", "metal", "none", "NO-READBACK")}
        if said != {vendor}:
            raise SystemExit(f"pack_wheel: {adir}/readback.txt says {sorted(said)}, "
                             f"directory says {vendor}; refusing to pack a mislabeled set")
        # THE ARCHITECTURE IS VERIFIED THE SAME WAY THE VENDOR IS: read back
        # from the binaries on the box (build_sets.sh), never typed. A set
        # whose read-back disagrees with its directory name is refused, the
        # exact failure mode that shipped 0.3.0 as sm_90a-only.
        ab = (adir / "arch_readback.txt").read_text().split()
        said_arch = {w for w in ab if ARCH_RE.match(w) or "," in w}
        if said_arch != {arch}:
            raise SystemExit(
                f"pack_wheel: {adir}/arch_readback.txt says {sorted(said_arch)}, "
                f"directory says {arch}; refusing to pack a mislabeled set")
        files = {}
        for tier in TIERS:
            d = adir if tier == "fast" else adir / tier
            for n in EXT_NAMES:
                so = d / (n + ".so")
                if not so.exists():
                    raise SystemExit(
                        f"pack_wheel: {vendor}/{arch} {tier} set is missing {n}.so")
                rel = (f"{vendor}/{arch}/{n}.so" if tier == "fast"
                       else f"{vendor}/{arch}/{tier}/{n}.so")
                files[rel] = so
        libs = {p.name: p for p in sorted((adir / ".libs").glob("*"))}
        if not libs:
            raise SystemExit(f"pack_wheel: {adir}/.libs is empty; stage_libs.py did not run")
        out.append((vendor, arch, files, libs, manifest))
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True,
                    help="a sets/<vendor> directory from build_sets.sh; give both")
    ap.add_argument("--out", default=str(PY_DIR / "dist"))
    ap.add_argument("--plat", default="linux_x86_64")
    ap.add_argument("--check-against", default="",
                    help="a macOS wheel whose METADATA must match this one's")
    a = ap.parse_args()

    proj = tomllib.loads((PY_DIR / "pyproject.toml").read_text())["project"]
    version = read_version()
    if proj["version"] != version:
        raise SystemExit(f"pack_wheel: pyproject says {proj['version']}, "
                         f"_version.py says {version}")
    readme = (REPO / "README.md").read_text()

    sets = [t for s in a.set for t in load_set(s)]
    keys = [(v, arch) for v, arch, _, _, _ in sets]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"pack_wheel: the same (vendor, arch) given twice: {keys}")

    # .libs layout: ONE shared mojolearn/.libs when every closure across
    # every (vendor, arch) set matches by name AND sha256 (2026-08-30
    # measured the two vendors' closures byte-identical); otherwise one per
    # vendor, which requires that vendor's architectures to agree among
    # themselves -- the MAX runtime does not vary by GPU architecture, so a
    # disagreement there is a build defect, refused rather than laid out.
    lib_sha = {k: {n: sha(p) for n, p in libs.items()}
               for k, (_, _, _, libs, _) in zip(keys, sets)}
    first = keys[0]
    shared = all(lib_sha[k] == lib_sha[first] for k in keys)
    if not shared:
        for vendor in {v for v, _ in keys}:
            ks = [k for k in keys if k[0] == vendor]
            if any(lib_sha[k] != lib_sha[ks[0]] for k in ks):
                raise SystemExit(
                    f"pack_wheel: the {vendor} architectures disagree on the "
                    "MAX runtime closure; the runtime does not vary by GPU "
                    "architecture, so one of these sets is broken: "
                    f"{[k[1] for k in ks]}")
    entries = {}  # archive path -> filesystem path
    entries["mojolearn_diagnostics.py"] = PY_DIR / "mojolearn_diagnostics.py"
    for py in sorted(PKG.glob("*.py")):
        entries[f"mojolearn/{py.name}"] = py
    seen_vendor_libs = set()
    for vendor, arch, files, libs, _ in sets:
        for rel, p in files.items():
            entries[f"mojolearn/{rel}"] = p
        if not shared and vendor not in seen_vendor_libs:
            seen_vendor_libs.add(vendor)
            for n, p in libs.items():
                entries[f"mojolearn/{vendor}/.libs/{n}"] = p
    if shared:
        for n, p in sets[0][3].items():
            entries[f"mojolearn/.libs/{n}"] = p

    dist = f"mojolearn-{version}.dist-info"
    tag = f"py3-none-{a.plat}"
    generated = {
        f"{dist}/METADATA": metadata_text(proj, readme).encode(),
        f"{dist}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: mojolearn pack_wheel.py\n"
            f"Root-Is-Purelib: false\nTag: {tag}\n").encode(),
        f"{dist}/entry_points.txt": (
            "[console_scripts]\n" + "".join(
                f"{k} = {v}\n" for k, v in proj.get("scripts", {}).items())).encode(),
        f"{dist}/top_level.txt": b"mojolearn\nmojolearn_diagnostics\n",
    }
    for lf in proj.get("license-files", []):
        generated[f"{dist}/licenses/{lf}"] = (REPO / lf).read_bytes()

    # THE CITATION SHIPS TOO, and it is generated here rather than relied on
    # from the tree. This packer writes the Linux wheel itself instead of
    # going through setuptools, so `[tool.setuptools.package-data]`'s
    # `CITATION.cff` entry -- which is what puts it in the macOS wheel -- has
    # no effect on this path. The 0.3.1 wheel carried licenses/LICENSE and
    # licenses/NOTICE and no machine-readable citation at all; a `pip
    # install` gave a user the licence and no way to cite the work. The
    # `[project.urls]` DOI and Citation entries need no help here, because
    # METADATA is generated from `proj` a few lines above.
    generated["mojolearn/CITATION.cff"] = (REPO / "CITATION.cff").read_bytes()

    if a.check_against:
        with zipfile.ZipFile(a.check_against) as z:
            names = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
            theirs = z.read(names[0]).decode()
        ours = generated[f"{dist}/METADATA"].decode()
        if theirs != ours:
            import difflib
            print("METADATA differs from", a.check_against)
            for ln in difflib.unified_diff(theirs.splitlines(), ours.splitlines(),
                                           "macos", "linux", lineterm="", n=0):
                print("  " + ln)
            print("  (a Linux classifier or a version bump is an expected line;"
                  " anything else is drift)")

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    whl = out / f"mojolearn-{version}-{tag}.whl"
    record = []
    with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in entries.items():
            data = src.read_bytes()
            z.writestr(arc, data)
            record.append(f"{arc},sha256={urlsafe_b64(hashlib.sha256(data).digest())},{len(data)}")
        for arc, data in generated.items():
            z.writestr(arc, data)
            record.append(f"{arc},sha256={urlsafe_b64(hashlib.sha256(data).digest())},{len(data)}")
        record.append(f"{dist}/RECORD,,")
        z.writestr(f"{dist}/RECORD", "\n".join(record) + "\n")

    size = whl.stat().st_size
    per_set = {}
    for vendor, arch, files, libs, manifest in sets:
        per_set[f"{vendor}/{arch}"] = {
            "extensions_bytes": manifest["bytes_extensions"],
            "runtime_libs_bytes": manifest["bytes_staged_libs"],
            "driver_libs_not_staged": manifest["driver_libs_not_staged"],
        }
    sizes = {
        "wheel": str(whl), "compressed_bytes": size,
        "compressed_mb": round(size / 1e6, 2),
        "pypi_limit_bytes": PYPI_LIMIT, "over_limit": size > PYPI_LIMIT,
        "libs_layout": "shared mojolearn/.libs" if shared else "per-vendor <vendor>/.libs",
        "sets": per_set, "tag": tag,
    }
    (out / f"SIZES-{version}-linux.json").write_text(json.dumps(sizes, indent=2))
    print(json.dumps(sizes, indent=2))
    if size > PYPI_LIMIT:
        print(f"\nOVER PyPI's {PYPI_LIMIT/1e6:.0f} MB LIMIT. STOP. Report the numbers "
              "above; do not split the name without them.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
