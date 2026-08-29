#!/usr/bin/env python3
"""Pack the fetched CUDA and HIP sets into ONE Linux wheel. Pure Python.

    python3 packaging/linux/pack_wheel.py \\
        --set bench/results/wheels/<stamp>-nvidia/sets/cuda \\
        --set bench/results/wheels/<stamp>-amd/sets/hip \\
        --out python/dist

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
    path = pathlib.Path(path).resolve()
    vendor = path.name
    if vendor not in LINUX_VENDORS:
        raise SystemExit(f"pack_wheel: set directory must be named cuda or hip: {path}")
    manifest = json.loads((path / "manifest.json").read_text())
    rb = (path / "readback.txt").read_text().split()
    said = {w for w in rb if w in ("cuda", "hip", "metal", "none", "NO-READBACK")}
    if said != {vendor}:
        raise SystemExit(f"pack_wheel: {path}/readback.txt says {sorted(said)}, "
                         f"directory says {vendor}; refusing to pack a mislabeled set")
    files = {}
    for tier in TIERS:
        d = path if tier == "fast" else path / tier
        for n in EXT_NAMES:
            so = d / (n + ".so")
            if not so.exists():
                raise SystemExit(f"pack_wheel: {vendor} {tier} set is missing {n}.so")
            rel = f"{vendor}/{n}.so" if tier == "fast" else f"{vendor}/{tier}/{n}.so"
            files[rel] = so
    libs = {p.name: p for p in sorted((path / ".libs").glob("*"))}
    if not libs:
        raise SystemExit(f"pack_wheel: {path}/.libs is empty; stage_libs.py did not run")
    return vendor, files, libs, manifest


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

    sets = [load_set(s) for s in a.set]
    vendors = [s[0] for s in sets]
    if len(set(vendors)) != len(vendors):
        raise SystemExit(f"pack_wheel: the same vendor given twice: {vendors}")

    # .libs layout: shared when every library matches by name AND sha256.
    lib_sha = {v: {n: sha(p) for n, p in libs.items()} for v, _, libs, _ in sets}
    shared = len(sets) > 1 and all(lib_sha[v] == lib_sha[vendors[0]] for v in vendors)
    entries = {}  # archive path -> filesystem path
    entries["mojolearn_diagnostics.py"] = PY_DIR / "mojolearn_diagnostics.py"
    for py in sorted(PKG.glob("*.py")):
        entries[f"mojolearn/{py.name}"] = py
    for vendor, files, libs, _ in sets:
        for rel, p in files.items():
            entries[f"mojolearn/{rel}"] = p
        if not shared:
            for n, p in libs.items():
                entries[f"mojolearn/{vendor}/.libs/{n}"] = p
    if shared:
        for n, p in sets[0][2].items():
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
    for vendor, files, libs, manifest in sets:
        per_set[vendor] = {
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
