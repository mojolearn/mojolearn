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
# FIFTEEN, and this is the FOURTH list that has to move when a binding is
# added. `packaging/linux/build_sets.sh` says "THREE lists move together" and
# names itself, its own EXT_NAMES, and the macOS pair -- it does not name this
# one, and this one is the list that decides what actually goes INTO the
# wheel.
#
# IT SHIPPED WRONG. The 0.4.0 Linux wheel built 2026-09-02 carries
# `_mamba_impl.py` and `_transformer_impl.py` and NO `.so` behind either, on
# all six architecture sets, because those two names were missing here. The
# legs built them -- every readback.txt lists `_mojolearn_mamba` and
# `_mojolearn_transformer` for cuda and hip in all three tiers, and every leg
# reports set_fast_count=15 -- and the pack loop below only raises on a name
# IN this tuple that is missing on disk, so a name absent from the tuple is
# never looked for and never missed.
#
# That is exactly the failure `build_sets.sh` warns about in words: "A binding
# missing from them is not a build error -- it is a wheel that ships without
# that extension and imports fine until the user touches the missing surface."
# `Mamba1Block` and `TransformerBlock` are both in `__all__`, so a Linux user
# of that wheel gets an ImportError naming a build command they cannot run.
EXT_NAMES = (
    "_mojolearn", "_mojolearn_gbdt", "_mojolearn_estimators", "_mojolearn_rf",
    "_mojolearn_trees", "_mojolearn_svm", "_mojolearn_solver",
    "_mojolearn_metrics", "_mojolearn_tsa", "_mojolearn_linalg",
    "_mojolearn_arima", "_mojolearn_training", "_mojolearn_gp",
    "_mojolearn_mamba", "_mojolearn_transformer",
)
TIERS = ("fast", "deterministic", "identical")
ARCH_RE = re.compile(r"^(sm_[0-9]+a?|gfx[0-9a-f]+)$")
PYPI_LIMIT = 100 * 1024 * 1024
LINUX_VENDORS = ("cuda", "hip")
RELEASE_061_SETS = {("cuda", "sm_89"), ("cuda", "sm_90"), ("hip", "gfx942")}


def release_inventory(sets, proof_paths, version, source_root=REPO):
    """Bind the explicit 0.6.1 payload to complete per-architecture builds.

    File inspection only. Build provenance is not installed/runtime admission.
    Byte-LM is required only in IDENTICAL; legacy generic sets remain unchanged.
    """
    keys = [(v, a) for v, a, _, _, _ in sets]
    if version != '0.6.1' or len(keys) != 3 or set(keys) != RELEASE_061_SETS:
        raise SystemExit('release-0.6.1 requires exactly CUDA sm_89/sm_90 and HIP gfx942')
    if len(proof_paths) != 3:
        raise SystemExit('release-0.6.1 requires three complete architecture build proofs')
    payload = {f'mojolearn/{rel}': sha(path).hex()
               for _, _, files, _, _ in sets for rel, path in files.items()}
    proofs, inventories, commits = {}, [], set()
    for path in proof_paths:
        raw = pathlib.Path(path).read_bytes()
        proof = json.loads(raw)
        if (proof.get('schema') != 'mojolearn.linux.build-provenance.v1'
                or proof.get('complete') is not True or proof.get('build_exit') != 0
                or proof.get('action') != 'build'):
            raise SystemExit('Incomplete architecture build proof')
        covered = {key for key in keys if any(
            name.startswith(f'mojolearn/{key[0]}/{key[1]}/')
            for name in proof.get('extensions', {}))}
        if len(covered) != 1:
            raise SystemExit('Each proof must cover exactly one advertised architecture')
        key = next(iter(covered))
        expected = {n: h for n, h in payload.items()
                    if n.startswith(f'mojolearn/{key[0]}/{key[1]}/')}
        required = {f'mojolearn/{key[0]}/{key[1]}/' +
                    ('' if mode == 'fast' else mode + '/') + name + '.so'
                    for mode in TIERS for name in (*EXT_NAMES, *(
                        ('_mojolearn_byte_lm',) if mode == 'identical' else ()))}
        if key in proofs or proof['extensions'] != expected or set(expected) != required:
            raise SystemExit('Duplicate, stale or incomplete architecture proof')
        inventory = proof['source_inventory']
        if (not inventory or len(inventory) != len({p for p, _ in inventory})
                or proof['source_sha256'] != hashlib.sha256(
                    json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()):
            raise SystemExit('Invalid build source inventory')
        for rel, digest in inventory:
            name = pathlib.PurePosixPath(rel)
            if name.is_absolute() or '..' in name.parts:
                raise SystemExit('Unsafe build source path')
            source = pathlib.Path(source_root) / rel
            if not source.resolve().is_relative_to(pathlib.Path(source_root).resolve()):
                raise SystemExit('Build source escapes source root')
            if sha(source).hex() != digest:
                raise SystemExit('Current source differs from build: ' + rel)
        commit = proof.get('source_commit', '')
        if not re.fullmatch('[0-9a-f]{40}', commit):
            raise SystemExit('Missing full build commit')
        commits.add(commit)
        inventories.append(inventory)
        proofs[key] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                           source_sha256=proof['source_sha256'])
    if len(commits) != 1 or any(i != inventories[0] for i in inventories[1:]):
        raise SystemExit('Architecture sets were built from different sources')
    return dict(schema='mojolearn.linux-payload.v1', version=version,
                release_profile='alpha-api', assembly_profile='release-0.6.1',
                source_commit=next(iter(commits)), source_inventory=inventories[0],
                sets={'/'.join(k): proofs[k] for k in sorted(proofs)},
                extensions=payload,
                optional_native={'_mojolearn_byte_lm': {
                    'included': True, 'supported_modes': ['identical'],
                    'unsupported_modes': ['fast', 'deterministic']}},
                qualification='Build and file provenance only; installed runtime and numerical checks required',
                runtime_coverage={ '/'.join(k): 'PENDING_INSTALLED_ARTIFACT' for k in sorted(proofs)})


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


def load_set(path, include_byte_lm=False):
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
        if include_byte_lm:
            expected_rows = {(tier, name) for tier in TIERS for name in
                             EXT_NAMES + (('_mojolearn_byte_lm',) if tier == 'identical' else ())}
            for witness, expected_value in (('readback.txt', vendor), ('arch_readback.txt', arch)):
                rows = [line.split() for line in (adir / witness).read_text().splitlines()]
                if (len(rows) != 46 or any(len(row) != 3 for row in rows)
                        or {(row[0], row[1]) for row in rows} != expected_rows
                        or any(row[2] != expected_value for row in rows)):
                    raise SystemExit(f'pack_wheel: incomplete release native readback in {adir / witness}')
        files = {}
        for tier in TIERS:
            d = adir if tier == "fast" else adir / tier
            names = EXT_NAMES + (('_mojolearn_byte_lm',)
                                 if include_byte_lm and tier == 'identical' else ())
            actual = {p.name for p in d.glob('_mojolearn*.so')}
            if actual != {n + '.so' for n in names}:
                raise SystemExit(f'pack_wheel: undeclared or missing native payload in {d}: {sorted(actual)}')
            for n in names:
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
    ap.add_argument('--profile', choices=('generic', 'release-0.6.1'), default='generic')
    ap.add_argument('--build-proof', action='append', default=[],
                    help='complete per-architecture build-provenance.json; three required for release-0.6.1')
    ap.add_argument("--check-against", default="",
                    help="a macOS wheel whose METADATA must match this one's")
    a = ap.parse_args()

    proj = tomllib.loads((PY_DIR / "pyproject.toml").read_text())["project"]
    version = read_version()
    if proj["version"] != version:
        raise SystemExit(f"pack_wheel: pyproject says {proj['version']}, "
                         f"_version.py says {version}")
    readme = (REPO / "README.md").read_text()

    sets = [t for s in a.set for t in load_set(s, include_byte_lm=a.profile == 'release-0.6.1')]
    keys = [(v, arch) for v, arch, _, _, _ in sets]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"pack_wheel: the same (vendor, arch) given twice: {keys}")
    if a.profile == 'generic' and a.build_proof:
        raise SystemExit('--build-proof requires an explicit release profile')
    inventory = (release_inventory(sets, a.build_proof, version)
                 if a.profile == 'release-0.6.1' else None)

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
    entries["mojolearn/ALPHA_API.md"] = PKG / "ALPHA_API.md"
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
    if inventory is not None:
        inventory['runtime_layout'] = 'shared' if shared else 'per-vendor'
        inventory['runtime_sha256'] = {
            name: sha(path).hex() for name, path in entries.items()
            if '/.libs/' in name}
        inventory['python_sha256'] = {
            name: sha(path).hex() for name, path in entries.items()
            if name.endswith('.py')}
        generated[f'{dist}/LINUX_PAYLOAD.json'] = (
            json.dumps(inventory, sort_keys=True, indent=2) + '\n').encode()
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
    if whl.exists():
        raise SystemExit('pack_wheel: refusing to overwrite existing artifact: ' + str(whl))
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
