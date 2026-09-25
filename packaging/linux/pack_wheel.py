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
`python/mojolearn/` and each explicitly declared subpackage (no tests), the identity payload the
macOS build copies in (since 0.8.6: `mojolearn/_identity_trace_diff.py`,
`mojolearn/_identity_break.py`, `mojolearn/reference_cards/`,
`mojolearn/identity_columns/<record>/` and its COMMIT witness), one copy of
every host binding under `mojolearn/host/`,
`mojolearn-<v>.dist-info/{METADATA,WHEEL,RECORD,entry_points.txt,
top_level.txt,licenses/LICENSE,licenses/NOTICE}`. METADATA is generated
from pyproject.toml with the same field order setuptools 84 wrote for
0.1.0, and `--check-against <macos wheel>` diffs the two METADATA bodies so
a drift is a visible line rather than a silent difference between the two
artifacts of one release.

REUSED BINDINGS (2026-09-23, tools/release_reuse.py). A set directory may
carry `reuse.json`: the bindings, host bindings and runtime libraries in it
that were TAKEN FROM THE LAST PUBLISHED WHEEL because their identity (source
closure, toolchain, flags, builder scripts, box image) had not changed. Every
file it names is verified here against the sha256 it records (which the
assembler verified against the published wheel's own payload), a set whose
bindings are all reused needs no read-back witnesses and no build proof (the
previous release's were its witnesses), and a set a leg built still needs
both. `--build-proof` is given once per set a leg built. LINUX_PAYLOAD.json
then records per binding whether it was built or reused and from which
release (`binding_origin`), and per set the same (`sets`).
"""

import argparse
import base64
import collections
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import zipfile


def _require_pkg_env():
    """Run under the pixi `pkg` environment or refuse BEFORE touching anything.

    2026-09-22 (0.8.14): `python3 packaging/linux/pack_wheel.py` under the
    system interpreter wrote the wheel, then died inside
    packaging/portable_math/stage.py on `No module named 'lief'` and deleted
    its own output. The finishing step needs lief (pinned in [feature.pkg]),
    so the packer checks for it first: when it is missing and pixi is on
    PATH it re-executes itself as `pixi run -e pkg python pack_wheel.py ...`
    (once, guarded by MOJOLEARN_PACK_WHEEL_REEXEC); otherwise it refuses
    with the command to use. `pixi run -e pkg pack-linux-wheel ...` is the
    documented spelling.
    """
    try:
        import lief  # noqa: F401
        import tomllib  # noqa: F401
        return
    except ImportError as exc:
        missing = exc.name or "lief"
    import shutil
    here = pathlib.Path(__file__).resolve()
    pixi = shutil.which("pixi")
    if pixi and os.environ.get("MOJOLEARN_PACK_WHEEL_REEXEC") != "1":
        print(f"pack_wheel: {missing} is not importable from {sys.executable}; "
              "re-running under `pixi run -e pkg python`", file=sys.stderr)
        env = dict(os.environ, MOJOLEARN_PACK_WHEEL_REEXEC="1")
        os.execve(pixi, [pixi, "run", "--manifest-path", str(here.parents[2] / "pixi.toml"),
                         "-e", "pkg", "python", str(here), *sys.argv[1:]], env)
    raise SystemExit(
        f"pack_wheel: {missing} is not importable from {sys.executable}. Run it "
        "under the pkg environment: pixi run -e pkg pack-linux-wheel <args> "
        "(or pixi run -e pkg python packaging/linux/pack_wheel.py <args>). "
        "Nothing was written.")


if __name__ == "__main__":
    _require_pkg_env()

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
    # The three TREE lanes: the only bindings with fast and deterministic
    # tiers (DEVIATION 2490, 2026-09-10).
    "_mojolearn_gbdt", "_mojolearn_rf", "_mojolearn_trees",
)
#: EVERYTHING ELSE BUILDS IDENTICAL ONLY (DEVIATION 2490). Held apart from
#: EXT_NAMES rather than removed: the exactness checks below compare the
#: files ON DISK against the expected set BOTH WAYS, so a name that is in no
#: list at all is neither required in identical nor refused in fast, which is
#: the miss the header above is about. `tier_names()` is the one place that
#: decides, and it is the same shape `build_sets.sh` uses. The reasoning and
#: the numbers are on `_TIERED` in python/mojolearn/_backend.py.
#: CLASSICAL ML (2026-09-25): fast and identical, never deterministic.
FAST_CLASSICAL_NAMES = (
    "_mojolearn", "_mojolearn_estimators", "_mojolearn_svm", "_mojolearn_solver",
    "_mojolearn_metrics", "_mojolearn_preprocessing", "_mojolearn_tsa",
    "_mojolearn_linalg", "_mojolearn_arima", "_mojolearn_gp",
    "_mojolearn_kernel_methods", "_mojolearn_mixture", "_mojolearn_hdbscan",
    "_mojolearn_resample", "_mojolearn_ivf",
)
IDENTICAL_ONLY_NAMES = (
    "_mojolearn_training", "_mojolearn_mamba",
    "_mojolearn_transformer", "_mojolearn_embedding",
)
TIERS = ("fast", "deterministic", "identical")


def tier_names(tier, include_byte_lm=False):
    """Every extension expected in `tier`, in pack order.

    Every binding but the three tree lanes, and the optional byte LM, exist
    in `identical` alone; a lower tier carries none of them. A set on disk that does not match this
    EXACTLY is refused, in both directions.
    """
    names = EXT_NAMES
    if tier in ("identical", "fast"):
        names = names + FAST_CLASSICAL_NAMES
    if tier == "identical":
        names = names + IDENTICAL_ONLY_NAMES
        if include_byte_lm:
            names = names + ("_mojolearn_byte_lm",)
    return names
ARCH_RE = re.compile(r"^(sm_[0-9]+a?|gfx[0-9a-f]+)$")
PYPI_LIMIT = 100 * 1024 * 1024
LINUX_VENDORS = ("cuda", "hip")
REUSE_SCHEMA = "mojolearn.linux.reused-bindings.v1"
#: One loaded sets/<vendor>/<arch> directory. `reuse` is None for a set a leg
#: built with nothing taken from the published wheel, else
#: dict(files={set-relative path: record}, origin={version, source_commit,
#: wheel, wheel_sha256}).
SetDir = collections.namedtuple("SetDir", "vendor arch files libs manifest hosts reuse")
# The combined three-architecture set. The name keeps the number the profile was
# authored under; the profile itself is RELEASE_PROFILE (`release-linux3`) and
# ships whatever version python/mojolearn/_version.py says. DEVIATION 2290.
RELEASE_061_SETS = {("cuda", "sm_89"), ("cuda", "sm_90"), ("hip", "gfx942")}
# DEVIATION 2293: the Hopper slot accepts sm_90 OR sm_90a, because the
# compiler decides which one it emits and it does not emit sm_90 on an H100.
# Asked for sm_90 with --target-accelerator, `mojo` produced 46 binaries all
# carrying sm_90a, and build_sets.sh refused the set rather than ship it under
# a name it was not verified to carry -- correctly, and that refusal is the
# only reason this was noticed rather than shipped.
#
# sm_90a is not a downgrade and not a workaround. `_backend.py` already treats
# it as the PREFERRED layout for this case: a device reporting sm_90 takes a
# carried sm_90a as "architecture-specific build for this exact device",
# ahead of any family fallback, because the `a` restricts WHICH DEVICES the
# code runs on and a Hopper device is the one it restricts to. Every sm_90
# device is Hopper, so no device loses the set by this name.
#
# What is NOT relaxed: exactly three sets, one per architecture slot, each
# still verified by reading the architecture back out of the binaries on the
# box. A set whose read-back disagrees with its directory is still refused.
RELEASE_HOPPER_ALTS = {("cuda", "sm_90"), ("cuda", "sm_90a")}
# DEVIATION 2290. ONE reader for the release version and ONE spelling of the
# profile name, owned by tools/verify_linux_surface_qualification.py and shared
# with the qualification checker and the alpha artifact verifier. This packer
# used to carry its own regex over _version.py and the literals '0.6.1' and
# 'release-0.6.1'; the version was never published under that number.
sys.path.append(str(REPO / "tools"))
from verify_linux_surface_qualification import (  # noqa: E402
    RELEASE_PROFILE, RELEASE_PROFILES, release_version, wheel_host_bindings)
# BY FILE PATH, not `from mojolearn import host_surface`: importing a
# submodule runs the package's __init__, which selects a backend and refuses
# on a box with no built binary (the release inventory test, Wheel CI).
# host_surface.py itself imports nothing from the package.
import importlib.util  # noqa: E402
_hs_spec = importlib.util.spec_from_file_location("mojolearn_host_surface", PY_DIR / "mojolearn" / "host_surface.py")
host_surface = importlib.util.module_from_spec(_hs_spec)
sys.modules[_hs_spec.name] = host_surface
_hs_spec.loader.exec_module(host_surface)


def release_inventory(sets, proof_paths, version, source_root=REPO):
    """Bind the explicit release-linux3 payload to complete per-architecture builds.

    File inspection only. Build provenance is not installed/runtime admission.
    Byte-LM is required only in IDENTICAL; legacy generic sets remain unchanged.
    `version` must be the version SOURCE_ROOT's _version.py declares
    (DEVIATION 2290); a literal never decides it.
    """
    keys = [(s.vendor, s.arch) for s in sets]
    keyset = set(keys)
    # DEVIATION 2293: normalise the Hopper slot before comparing, so sm_90 and
    # sm_90a are the same slot and neither can appear twice.
    hopper = keyset & RELEASE_HOPPER_ALTS
    normalised = (keyset - RELEASE_HOPPER_ALTS) | ({("cuda", "sm_90")} if hopper else set())
    if (version != read_version(source_root) or len(keys) != 3
            or len(hopper) > 1 or normalised != RELEASE_061_SETS):
        raise SystemExit(RELEASE_PROFILE + ' requires exactly CUDA sm_89, CUDA sm_90'
                         ' or sm_90a, and HIP gfx942')
    # A SET A LEG BUILT NEEDS ITS PROOF; a set whose every binding was taken
    # from the published wheel has none (its witnesses are the previous
    # release's), so the proofs number the built sets, not three.
    reused_of = {(s.vendor, s.arch): (s.reuse or {}).get('files', {}) for s in sets}
    # s.files is keyed vendor/arch/<set-relative>; reuse.json is set-relative
    built_keys = {(s.vendor, s.arch) for s in sets
                  if any(rel.split('/', 2)[2] not in reused_of[(s.vendor, s.arch)] for rel in s.files)}
    # A HOST BINDING BUILT FOR THIS RELEASE while every GPU set is the
    # published one: one leg (release_reuse's cheapest) built it and read it
    # back, so exactly one proof is required, that leg's.
    host_built = {n for s in sets for n in (s.hosts or {}) if f'host/{n}.so' not in reused_of[(s.vendor, s.arch)]}
    proofs_needed = len(built_keys) if built_keys else (1 if host_built else 0)
    if len(proof_paths) != proofs_needed:
        raise SystemExit(RELEASE_PROFILE + ' requires one complete build proof per set a leg built: '
                         f'{len(built_keys)} built ({", ".join("/".join(k) for k in sorted(built_keys)) or "none"}), '
                         f'{len(host_built)} host binding(s) built ({", ".join(sorted(host_built)) or "none"}), '
                         f'{proofs_needed} proof(s) needed, {len(proof_paths)} given')
    payload = {f'mojolearn/{rel}': sha(path).hex()
               for s in sets for rel, path in s.files.items()}
    origin = {}
    for s in sets:
        reused = reused_of[(s.vendor, s.arch)]
        for rel in s.files:
            rec = reused.get(rel.split('/', 2)[2])
            origin[f'mojolearn/{rel}'] = (
                dict(origin='reused', set=f'{s.vendor}/{s.arch}', from_release=s.reuse['origin'],
                     identity_digest=rec.get('identity_digest'), rebuilt_sha256=rec.get('rebuilt_sha256'))
                if rec else dict(origin='built', set=f'{s.vendor}/{s.arch}'))
    # The CPU training binding sits outside this map on purpose: every key here
    # is mojolearn/<vendor>/<arch>/..., and a vendor-neutral file belongs to no
    # architecture. It is recorded separately so the payload record still names
    # every shipped binary, with the legs that produced it and its digest.
    # Keyed by basename since 0.8.6: every host family the manifest ships,
    # each with the legs that built it and one digest (the byte compare in
    # main() has already refused legs that disagree).
    host_record = {}
    for name in HOST_NAMES:
        seen = {f'{s.vendor}/{s.arch}': sha(s.hosts[name]).hex() for s in sets if name in (s.hosts or {})}
        if seen:
            taken = [s for s in sets if f'host/{name}.so' in reused_of[(s.vendor, s.arch)]]
            host_record[name] = dict(archive_path=f'mojolearn/host/{name}.so',
                                     sha256=sorted(set(seen.values()))[0],
                                     vendor='cpu', supported_modes=['identical'],
                                     unsupported_modes=['fast', 'deterministic'],
                                     built_by=sorted(seen),
                                     origin='reused' if taken else 'built',
                                     from_release=taken[0].reuse['origin'] if taken else None,
                                     scope='CPU training and inference with no GPU; one '
                                           'copy for every architecture in this wheel')
            origin[f'mojolearn/host/{name}.so'] = dict(
                origin=host_record[name]['origin'], from_release=host_record[name]['from_release'],
                identity_digest=taken[0].reuse['files'][f'host/{name}.so'].get('identity_digest') if taken else None,
                built_by=[k for k in sorted(seen) if k not in {f'{s.vendor}/{s.arch}' for s in taken}])
    if set(host_record) != set(HOST_NAMES):
        raise SystemExit(RELEASE_PROFILE + ' requires every host binding the manifest ships; missing: '
                         + ', '.join(sorted(set(HOST_NAMES) - set(host_record))))
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
        if key not in built_keys and not host_built:
            raise SystemExit('A build proof was given for a set whose every binding is reused: ' + '/'.join(key))
        expected = {n: h for n, h in payload.items()
                    if n.startswith(f'mojolearn/{key[0]}/{key[1]}/')}
        required = {f'mojolearn/{key[0]}/{key[1]}/' +
                    ('' if mode == 'fast' else mode + '/') + name + '.so'
                    for mode in TIERS for name in tier_names(mode, True)}
        # The leg built every binding of its set; the proof must name them
        # all, and its digest must be the shipped digest for every binding
        # the release did not take from the published wheel. For the taken
        # ones the leg's digest is recorded beside the shipped one.
        prefix = f'mojolearn/{key[0]}/{key[1]}/'
        built = {n: h for n, h in expected.items() if n[len(prefix):] not in reused_of[key]}
        if (key in proofs or set(proof['extensions']) != required or set(expected) != required
                or {n: proof['extensions'][n] for n in built} != built):
            raise SystemExit('Duplicate, stale or incomplete architecture proof')
        for n in expected:
            if n not in built:
                origin[n]['rebuilt_sha256'] = proof['extensions'][n]
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
    if len(commits) > 1 or any(i != inventories[0] for i in inventories[1:]):
        raise SystemExit('Architecture sets were built from different sources')
    if commits:
        commit, inventory = next(iter(commits)), inventories[0]
    else:
        # Nothing was built: the wheel's source witness is the checkout being
        # packed, its native inventory computed here exactly as a leg would
        # have (tracked native source only).
        from check_linux_release_qualification import tracked_native_inventory
        commit = subprocess.run(['git', '-C', str(source_root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        if not re.fullmatch('[0-9a-f]{40}', commit):
            raise SystemExit('No build proof and no git commit to witness the reused wheel')
        inventory = tracked_native_inventory(source_root)
    set_records = {}
    for s in sets:
        key = (s.vendor, s.arch)
        if key in proofs:
            set_records['/'.join(key)] = dict(proofs[key], origin='built' if not reused_of[key] else 'mixed',
                                              reused=sorted(reused_of[key]))
        else:
            set_records['/'.join(key)] = dict(origin='reused', from_release=s.reuse['origin'],
                                              reused=sorted(reused_of[key]))
    n_reused = sum(1 for o in origin.values() if o['origin'] == 'reused')
    return dict(schema='mojolearn.linux-payload.v1', version=version,
                release_profile='alpha-api', assembly_profile=RELEASE_PROFILE,
                source_commit=commit, source_inventory=inventory,
                sets=set_records,
                extensions=payload,
                binding_origin=origin,
                reuse=dict(built=len(origin) - n_reused, reused=n_reused,
                           from_release=next((s.reuse['origin'] for s in sets if s.reuse), None)),
                optional_native={**{n: {
                    'included': True, 'supported_modes': ['identical'],
                    'unsupported_modes': ['fast', 'deterministic']}
                    for n in ('_mojolearn_byte_lm',) + tuple(host_record) + IDENTICAL_ONLY_NAMES},
                    **{n: {
                    'included': True, 'supported_modes': ['fast', 'identical'],
                    'unsupported_modes': ['deterministic']}
                    for n in FAST_CLASSICAL_NAMES}},
                # One record per host binding, keyed by basename, so the
                # payload says which host families this wheel carries rather
                # than leaving a reader to infer it from the archive.
                host_native=host_record,
                qualification='Build and file provenance only; installed runtime and numerical checks required',
                runtime_coverage={'/'.join(k): 'PENDING_INSTALLED_ARTIFACT' for k in sorted(keys)})


def urlsafe_b64(digest):
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def read_version(root=REPO):
    """`__version__` of ROOT/python/mojolearn/_version.py through the shared reader (DEVIATION 2290)."""
    try:
        return release_version(root)
    except (OSError, ValueError) as exc:
        raise SystemExit("pack_wheel: " + str(exc))


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


#: The host (CPU) bindings. Vendor-neutral and tier-neutral: one copy of each
#: per wheel under `mojolearn/host/`, which is where the runtime looks for
#: them (by path, or through `_backend.load_host_module`) rather than through
#: `_backend.binding()`. They are therefore not members of any tier list and
#: not members of any vendor directory, and every check that assumes "a
#: vendor binary inside a tier" exempts them by these names. WHICH names is
#: READ from python/mojolearn/host_surface.py (every family the manifest
#: declares since 0.8.6, the packaging lane of 2026-09-14); until 0.8.5 the
#: byte LM's was the only one and this file spelled it by hand.
HOST_NAMES = wheel_host_bindings()


def host_witnesses(set_paths):
    """name -> sha256 of every host binding some architecture directory under
    `set_paths` READ BACK as `cpu` (a `host <name> cpu` row in its
    readback.txt, the binary beside it). A host binding has no device code and
    the same bytes are placed into every set; a set that reuses every GPU
    binding from the published wheel has no read-back of its own, so a host
    binding built for this release is proven there by the leg that built and
    read it back, byte for byte (release_reuse's linux-wait requires the host
    bindings to be byte-identical across the legs)."""
    out = {}
    for sp in set_paths:
        sp = pathlib.Path(sp).resolve()
        for adir in sorted(d for d in sp.iterdir() if d.is_dir() and ARCH_RE.match(d.name)):
            rb = adir / "readback.txt"
            if not rb.is_file():
                continue
            for line in rb.read_text().splitlines():
                row = line.split()
                if len(row) == 3 and row[0] == "host" and row[2] == "cpu" and (adir / "host" / f"{row[1]}.so").is_file():
                    digest = sha(adir / "host" / f"{row[1]}.so")
                    if out.setdefault(row[1], digest) != digest:
                        raise SystemExit(f"pack_wheel: host binding {row[1]} was read back with different bytes in two sets; "
                                         "one of them is not this release's")
    return out


def load_set(path, include_byte_lm=False, host_witnesses_by_name=None):
    """Every (vendor, arch, files, libs, manifest) under one sets/<vendor>
    directory. One tuple per architecture subdirectory. `host_witnesses_by_name`
    (see `host_witnesses`) lets a set that reuses every GPU binding carry a
    host binding built for this release when a witnessed set holds the same
    bytes."""
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
        reuse = load_reuse(adir, vendor, arch)
        reused = reuse["files"] if reuse else {}
        expected_rels = {("" if tier == "fast" else tier + "/") + name + ".so"
                         for tier in TIERS for name in tier_names(tier, include_byte_lm)}
        all_reused = bool(reuse) and expected_rels <= set(reused) and (
            not include_byte_lm or all(f"host/{n}.so" in reused for n in HOST_NAMES))
        has_witnesses = (adir / "readback.txt").is_file() and (adir / "arch_readback.txt").is_file()
        witnessed_built = []
        if not has_witnesses and not all_reused:
            missing = expected_rels - set(reused)
            if include_byte_lm:
                missing |= {f"host/{n}.so" for n in HOST_NAMES if f"host/{n}.so" not in reused}
            built_host = {rel for rel in missing if rel.startswith("host/")}
            if not reuse or built_host != missing or not host_witnesses_by_name:
                raise SystemExit(
                    f"pack_wheel: {adir} has no readback.txt or arch_readback.txt and not every binding "
                    "in it is taken from the published wheel (reuse.json); a built binding needs its read-back")
            # EVERY GPU BINDING IS THE PUBLISHED ONE; the host bindings built
            # for this release were read back as `cpu` by the leg that built
            # them, in a set that holds the same bytes.
            for rel in sorted(built_host):
                name = rel[len("host/"):-len(".so")]
                so = adir / rel
                if not so.is_file() or host_witnesses_by_name.get(name) != sha(so):
                    raise SystemExit(
                        f"pack_wheel: {adir}/{rel} was built for this release and no set with a read-back holds "
                        "the same bytes; a built binding needs its read-back")
                witnessed_built.append(name)
        if not has_witnesses:
            # EVERY GPU BINDING WAS TAKEN FROM THE PUBLISHED WHEEL. Its vendor and
            # architecture were read back on the box that built it, for the
            # release that shipped it; the bytes verified above are those. A
            # host binding built for this release is witnessed in another set.
            host_named = [n for n in HOST_NAMES if f"host/{n}.so" in reused or n in witnessed_built]
        else:
            # THE HOST ROWS ARE READ SEPARATELY, and by name. The CPU training
            # binding has no device code and answers 'cpu', so folding it into the
            # vendor and architecture comparisons below would either refuse a
            # correct set or force those comparisons to accept 'cpu' from a GPU
            # binary, which is the failure they exist to catch. Rows whose first
            # field is `host` are pulled out here and checked on their own terms.
            rb_lines = [ln.split() for ln in (adir / "readback.txt").read_text().splitlines()]
            host_rows = [r for r in rb_lines if r and r[0] == "host"]
            rb = [w for r in rb_lines if not (r and r[0] == "host") for w in r]
            said = {w for w in rb if w in ("cuda", "hip", "metal", "none", "NO-READBACK")}
            if said != {vendor}:
                raise SystemExit(f"pack_wheel: {adir}/readback.txt says {sorted(said)}, "
                                 f"directory says {vendor}; refusing to pack a mislabeled set")
            if any(len(r) != 3 or r[2] != "cpu" for r in host_rows):
                raise SystemExit(
                    f"pack_wheel: {adir}/readback.txt host row is not 'cpu': {host_rows}. "
                    "A GPU vendor there means the CPU-only build saw an accelerator target.")
            host_named = [r[1] for r in host_rows]
            if any(n not in HOST_NAMES for n in host_named) or len(set(host_named)) != len(host_named):
                raise SystemExit(
                    f"pack_wheel: {adir}/readback.txt names host bindings the manifest does not ship, "
                    f"or one twice: {host_named}; the manifest ships {list(HOST_NAMES)}")
            if include_byte_lm and set(host_named) != set(HOST_NAMES):
                raise SystemExit(
                    f"pack_wheel: {adir}/readback.txt names {sorted(host_named)}; the release profile "
                    f"requires every host binding the manifest ships: {list(HOST_NAMES)}")
            # THE ARCHITECTURE IS VERIFIED THE SAME WAY THE VENDOR IS: read back
            # from the binaries on the box (build_sets.sh), never typed. A set
            # whose read-back disagrees with its directory name is refused, the
            # exact failure mode that shipped 0.3.0 as sm_90a-only.
            ab_lines = [ln.split() for ln in (adir / "arch_readback.txt").read_text().splitlines()]
            host_arch_rows = [r for r in ab_lines if r and r[0] == "host"]
            ab = [w for r in ab_lines if not (r and r[0] == "host") for w in r]
            said_arch = {w for w in ab if ARCH_RE.match(w) or "," in w}
            if said_arch != {arch}:
                raise SystemExit(
                    f"pack_wheel: {adir}/arch_readback.txt says {sorted(said_arch)}, "
                    f"directory says {arch}; refusing to pack a mislabeled set")
            if any(len(r) != 3 or r[2] != "NONE-BY-DESIGN" for r in host_arch_rows):
                raise SystemExit(
                    f"pack_wheel: {adir}/arch_readback.txt host row names architectures: "
                    f"{host_arch_rows}. A host binding must carry no device code.")
            if sorted(r[1] for r in host_arch_rows) != sorted(host_named):
                raise SystemExit(
                    f"pack_wheel: {adir}/arch_readback.txt and readback.txt name different host bindings")
            if include_byte_lm:
                expected_rows = {(tier, name) for tier in TIERS
                                 for name in tier_names(tier, True)}
                for witness, expected_value in (('readback.txt', vendor), ('arch_readback.txt', arch)):
                    rows = [line.split() for line in (adir / witness).read_text().splitlines()
                            if not line.startswith('host ')]
                    if (len(rows) != len(expected_rows) or any(len(row) != 3 for row in rows)
                            or {(row[0], row[1]) for row in rows} != expected_rows
                            or any(row[2] != expected_value for row in rows)):
                        raise SystemExit(f'pack_wheel: incomplete release native readback in {adir / witness}')
        # OPTIONAL WHEN ABSENT (generic profile only), because a set built
        # before this payload existed is still a valid set and a rebuild of an
        # older commit must not be refused. A named binding must be on disk;
        # a file on disk that no row names is refused, because an unread
        # binary is one whose vendor and column nobody checked. `host_payload`
        # is a dict basename -> path, empty when the set carries none, and
        # `main` says so once rather than shipping an export that cannot work.
        host_payload = {}
        for name in host_named:
            host_so = adir / "host" / f"{name}.so"
            if not host_so.exists():
                raise SystemExit(
                    f"pack_wheel: {adir}/readback.txt names {name} but {host_so} is absent")
            host_payload[name] = host_so
        on_disk = {p.name[:-3] for p in (adir / "host").glob("_mojolearn_*_host.so")} if (adir / "host").is_dir() else set()
        if on_disk - set(host_named):
            raise SystemExit(
                f"pack_wheel: {adir}/host/ holds bindings readback.txt never read back: "
                f"{sorted(on_disk - set(host_named))}")
        files = {}
        for tier in TIERS:
            d = adir if tier == "fast" else adir / tier
            names = tier_names(tier, include_byte_lm)
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
        if reuse:
            # `files` is keyed vendor/arch/<set-relative>; reuse.json is set-relative
            packed = {rel.split("/", 2)[2] for rel in files}
            unknown = set(reused) - packed - {f"host/{n}.so" for n in host_named} - {f".libs/{n}" for n in libs}
            if unknown:
                raise SystemExit(f"pack_wheel: {adir}/reuse.json names files this set does not pack: {sorted(unknown)}")
        out.append(SetDir(vendor, arch, files, libs, manifest, host_payload, reuse))
    return out


def load_reuse(adir, vendor, arch):
    """`reuse.json` of one set directory, every file it names verified
    against the published bytes' sha256 it records; None when absent. A file
    that differs is refused: a reused binding is the published one or it is
    not reused."""
    path = adir / "reuse.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    origin = doc.get("from_release") or {}
    if (doc.get("schema") != REUSE_SCHEMA or doc.get("set") != f"{vendor}/{arch}"
            or not isinstance(doc.get("files"), dict)
            or not all(origin.get(k) for k in ("version", "source_commit", "wheel", "wheel_sha256"))):
        raise SystemExit(f"pack_wheel: {path} is not a {REUSE_SCHEMA} record for {vendor}/{arch}")
    for rel, rec in sorted(doc["files"].items()):
        p = adir / rel
        if pathlib.PurePosixPath(rel).is_absolute() or ".." in rel.split("/"):
            raise SystemExit(f"pack_wheel: {path} names an unsafe path: {rel}")
        if not p.is_file():
            raise SystemExit(f"pack_wheel: {path} names {rel}, which is absent from {adir}")
        want = rec.get("sha256") if isinstance(rec, dict) else None
        if not isinstance(want, str) or not re.fullmatch("[0-9a-f]{64}", want):
            raise SystemExit(f"pack_wheel: {path} records no sha256 for {rel}")
        if sha(p).hex() != want:
            raise SystemExit(
                f"pack_wheel: {adir}/{rel} differs from the published {origin['version']} bytes reuse.json "
                f"records ({want[:12]}); a reused binding must be the published one, refusing to pack it")
    return dict(files=doc["files"], origin=dict(version=origin["version"], source_commit=origin["source_commit"],
                                                wheel=origin["wheel"], wheel_sha256=origin["wheel_sha256"]))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def require_shipped_python_at_commit(entries, commit, repo=REPO):
    """Every shipped .py read from REPO must be byte-identical to COMMIT's copy.

    The build proofs bind native source only, and their inventory leaves out
    python/mojolearn/tests/, which never ships. This is the check that the
    wheel's Python equals the built commit: an uncommitted edit, or an ignored
    or untracked module sitting in a packaged directory, is refused here.
    Git blob ids are computed locally, so no file is read through git.
    """
    import subprocess
    repo = pathlib.Path(repo).resolve()
    try:
        listing = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", "--full-tree", commit],
                                 check=True, capture_output=True, timeout=60).stdout.decode()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"pack_wheel: cannot list build commit {commit} in {repo}: {exc}")
    blobs = {}
    for row in filter(None, listing.split("\0")):
        meta, path = row.split("\t", 1)
        blobs[path] = meta.split()[2]
    for arcname, source in sorted(entries.items()):
        if not arcname.endswith(".py"):
            continue
        source = pathlib.Path(source).resolve()
        if not source.is_relative_to(repo):
            raise SystemExit(f"pack_wheel: {arcname} is read from outside the checkout: {source}")
        rel = source.relative_to(repo).as_posix()
        data = source.read_bytes()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blobs.get(rel) != blob:
            state = "is not in" if rel not in blobs else "differs from"
            raise SystemExit(f"pack_wheel: shipped {arcname} ({rel}) {state} build commit {commit}")


def python_package_entries():
    """Use the same explicit Python package inventory as the macOS wheel."""
    config = tomllib.loads((PY_DIR / "pyproject.toml").read_text())
    entries = {}
    for package in config["tool"]["setuptools"]["packages"]:
        directory = PY_DIR.joinpath(*package.split("."))
        for source in sorted(directory.glob("*.py")):
            entries[source.relative_to(PY_DIR).as_posix()] = source
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True,
                    help="a sets/<vendor> directory from build_sets.sh; give both")
    ap.add_argument("--out", default=str(PY_DIR / "dist"))
    ap.add_argument("--plat", default="linux_x86_64")
    # DEVIATION 2290: `release-0.6.1` is the deprecated alias of release-linux3.
    ap.add_argument('--profile', choices=('generic', RELEASE_PROFILE, 'release-0.6.1'), default='generic')
    ap.add_argument('--build-proof', action='append', default=[],
                    help='complete per-architecture build-provenance.json; three required for ' + RELEASE_PROFILE)
    ap.add_argument("--check-against", default="",
                    help="a macOS wheel whose METADATA must match this one's")
    ap.add_argument("--portable-math-helper", type=pathlib.Path,
                    help="Linux-built libMojolearnMath.so; required when packing on macOS")
    a = ap.parse_args()
    if a.profile in RELEASE_PROFILES:
        a.profile = RELEASE_PROFILE  # DEVIATION 2290: the alias maps to the same path

    proj = tomllib.loads((PY_DIR / "pyproject.toml").read_text())["project"]
    version = read_version()
    if proj["version"] != version:
        raise SystemExit(f"pack_wheel: pyproject says {proj['version']}, "
                         f"_version.py says {version}")
    readme = (REPO / "README.md").read_text()

    witnesses = host_witnesses(a.set)
    sets = [t for s in a.set for t in load_set(s, include_byte_lm=a.profile == RELEASE_PROFILE, host_witnesses_by_name=witnesses)]
    keys = [(s.vendor, s.arch) for s in sets]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"pack_wheel: the same (vendor, arch) given twice: {keys}")
    if a.profile == 'generic' and a.build_proof:
        raise SystemExit('--build-proof requires an explicit release profile')
    if a.profile != RELEASE_PROFILE and any(s.reuse for s in sets):
        raise SystemExit('pack_wheel: reuse.json (bindings taken from a published wheel) needs the release profile')
    inventory = (release_inventory(sets, a.build_proof, version)
                 if a.profile == RELEASE_PROFILE else None)
    if inventory is not None:
        r = inventory['reuse']
        print(f"pack_wheel: {r['built']} binding(s) built by the legs, {r['reused']} taken from "
              + (f"{r['from_release']['version']} ({r['from_release']['wheel']}, sha256 "
                 f"{r['from_release']['wheel_sha256'][:12]})" if r['from_release'] else 'no previous release'))
        for key, rec in sorted(inventory['sets'].items()):
            print(f"pack_wheel: set {key}: {rec['origin']}, {len(rec.get('reused', []))} file(s) reused")

    # .libs layout: ONE shared mojolearn/.libs when every closure across
    # every (vendor, arch) set matches by name AND sha256 (2026-08-30
    # measured the two vendors' closures byte-identical); otherwise one per
    # vendor, which requires that vendor's architectures to agree among
    # themselves -- the MAX runtime does not vary by GPU architecture, so a
    # disagreement there is a build defect, refused rather than laid out.
    lib_sha = {k: {n: sha(p) for n, p in libs.items()}
               for k, libs in zip(keys, (s.libs for s in sets))}
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
    entries.update(python_package_entries())
    entries["mojolearn/ALPHA_API.md"] = PKG / "ALPHA_API.md"
    entries["mojolearn/Hendel_2026_bitwise_identical_gpu_ml_preprint.pdf"] = PKG / "Hendel_2026_bitwise_identical_gpu_ml_preprint.pdf"
    seen_vendor_libs = set()
    for vendor, arch, files, libs, _, _, _ in sets:
        for rel, p in files.items():
            entries[f"mojolearn/{rel}"] = p
        if not shared and vendor not in seen_vendor_libs:
            seen_vendor_libs.add(vendor)
            for n, p in libs.items():
                entries[f"mojolearn/{vendor}/.libs/{n}"] = p
    if shared:
        for n, p in sets[0].libs.items():
            entries[f"mojolearn/.libs/{n}"] = p

    # ONE COPY OF EACH, AND EVERY SET THAT CARRIES ONE MUST CARRY THE SAME
    # BYTES. A host binding is vendor-neutral, so each architecture leg builds
    # its own and the wheel ships exactly one. Differing bytes across legs
    # means they were not built from one source (the 0.8.5 freeze caught the
    # byte LM's copies differing by 43 bytes because a detected-column
    # read-back folded the build box's GPU name into a vendor-neutral binary),
    # which is the same defect the MAX runtime closure check above refuses, so
    # it is refused here too rather than resolved by picking a winner. Absent
    # everywhere is legal for the generic profile and says so once: a wheel
    # without a binding has no CPU path for that family and must not pretend
    # to. The release profile requires every binding the manifest ships, in
    # every set (load_set refused a short readback.txt already).
    carried = 0
    for name in HOST_NAMES:
        host_payloads = {(s.vendor, s.arch): s.hosts[name] for s in sets if name in (s.hosts or {})}
        if not host_payloads:
            print(f"pack_wheel: NO {name} in any set; this wheel has no CPU path for that family")
            continue
        digests = {k: sha(p).hex() for k, p in host_payloads.items()}
        if len(set(digests.values())) != 1:
            raise SystemExit(
                f"pack_wheel: the architecture legs disagree on {name}; it is "
                "vendor-neutral, so one copy is wrong: "
                + ", ".join(f"{v}/{a}={d[:12]}" for (v, a), d in sorted(digests.items())))
        if len(host_payloads) != len(sets):
            raise SystemExit(
                f"pack_wheel: {name} is carried by {len(host_payloads)} of {len(sets)} sets; "
                "every leg builds every host binding or none does")
        entries[f"mojolearn/host/{name}.so"] = next(iter(host_payloads.values()))
        carried += 1
        print(f"pack_wheel: {name} carried once from "
              f"{'/'.join(next(iter(host_payloads)))}, sha256 {next(iter(digests.values()))[:12]}, "
              f"byte-identical across {len(host_payloads)} legs")
    print(f"pack_wheel: {carried} of {len(HOST_NAMES)} host bindings carried")

    # THE IDENTITY PAYLOAD (the packaging lane, 2026-09-14; docs/VERIFY.md),
    # the same files packaging/macos/build_release_wheel.sh copies into the
    # package tree: the one card comparator and the reference card directory
    # for `python -m mojolearn verify`, and tools/identity_break.py, the three
    # training GPU columns the manifest names and a commit witness for
    # `python -m mojolearn identity`. Read straight from their single sources
    # in this checkout; the wheel has no second implementation of anything.
    entries["mojolearn/_identity_trace_diff.py"] = REPO / "tools" / "identity_trace_diff.py"
    entries["mojolearn/_identity_break.py"] = REPO / "tools" / "identity_break.py"
    for card in sorted((PKG / "reference_cards").iterdir()):
        if card.is_file():
            entries[f"mojolearn/reference_cards/{card.name}"] = card
    # `python -m mojolearn verify --all` (2026-09-15): the reference hash
    # table and the portable models, tracked files, the same globs as
    # pyproject.toml's package-data.
    verify_ref = PKG / "verify_reference"
    if not (verify_ref / "table.json").is_file():
        raise SystemExit("pack_wheel: python/mojolearn/verify_reference/table.json is missing")
    for src in sorted(verify_ref.glob("*.json")) + sorted((verify_ref / "models").glob("*")):
        if src.is_file():
            entries[f"mojolearn/{src.relative_to(PKG).as_posix()}"] = src
    # The shared stager validates every fixture digest before either platform packs it.
    from verification_ctr_payload import model_entries
    for name, src in model_entries(REPO).items():
        entries[f"mojolearn/verify_reference/ctr_models/{name}"] = src
    record_dir = host_surface.training_gpu_column_record()
    for col in host_surface.TRAINING_GPU_COLUMNS:
        src = REPO / col
        if not src.is_file():
            raise SystemExit(f"pack_wheel: the manifest names {col}, which is not in this checkout")
        entries[f"mojolearn/identity_columns/{record_dir}/{src.name}"] = src
    if inventory is not None:
        witness = inventory["source_commit"]
        require_shipped_python_at_commit(entries, witness)
    else:
        import subprocess
        witness = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
    if not re.fullmatch("[0-9a-f]{40}", witness or ""):
        raise SystemExit("pack_wheel: no commit witness for the identity columns "
                         "(no release proof and no git checkout)")

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
    generated["mojolearn/identity_columns/COMMIT"] = (witness + "\n").encode()

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

    # Finalize cached runtime closures too; never trust an older set to be libm-free.
    import subprocess
    command = [sys.executable, str(REPO / "packaging/portable_math/wheel.py"), str(whl)]
    if a.portable_math_helper:
        command += ["--helper", str(a.portable_math_helper)]
    try:
        subprocess.run(command, check=True)
    except Exception:
        whl.unlink(missing_ok=True)
        raise

    # Audit independently of the package allow-list above, so adding an API
    # without updating packaging fails at build time rather than after upload.
    from wheel_api_audit import audit
    surface = audit([whl])
    (out / f"API-{version}-linux.json").write_text(json.dumps(surface, indent=2) + "\n")
    if not surface['wheels'][0]['source_payload_complete']:
        whl.unlink()
        raise SystemExit('pack_wheel: incomplete source/API payload; see API report')

    size = whl.stat().st_size
    per_set = {}
    for vendor, arch, files, libs, manifest, _, _ in sets:
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
