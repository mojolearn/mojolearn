#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Pack the fetched CUDA and HIP sets into the Linux wheels. Pure Python.

    python3 packaging/linux/pack_wheel.py \\
        --set bench/results/wheels/<stamp1>-nvidia/sets/cuda \\
        --set bench/results/wheels/<stamp2>-nvidia/sets/cuda \\
        --set bench/results/wheels/<stamp3>-amd/sets/hip \\
        --out python/dist

FOUR WHEELS BY DEFAULT (python/mojolearn/gpu_plugins.py): core, NVIDIA
native sm_89, AMD gfx942 and NVIDIA PTX sm_80. The core requires both
vendors, and NVIDIA requires PTX, all pinned to the same version.
Default pip installation includes every released target.

`--wheels nvidia,nvidia-ptx` emits the native NVIDIA wheel and its exactly
pinned PTX dependency. The PTX slot (`cuda/sm_80`, installed at
`cuda_ptx/sm_80`) has its own build proof. Both package markers bind the
same PTX manifest SHA256; native and PTX files have disjoint owners.
Every wheel is checked against PyPI's 100 MiB file limit here, at pack time;
an oversize wheel fails the pack with its per-set compressed sizes.
Kernel bytes are unchanged. Native roots move to cuda_native/ and hip_native/
at the same directory depth, preserving RUNPATH while avoiding ownership overlap
with old vendor wheels during upgrades. Only the core carries the shared runtime.
Combined profiles retain their previous native layout.

EVERY SET CARRIES AN ARCHITECTURE LEVEL (2026-08-30): a `--set` directory is
`sets/<vendor>/` holding one or more `<arch>/` subdirectories (`sm_80`,
`gfx942`, ...), because one `mojo build` emits device code for exactly one
architecture on the native release path -- the sm_90a-only 0.3.0 wheel failed 27 of 29 lanes
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

THE COMBINED PROFILES' TAG IS `linux_x86_64`, DELIBERATELY (the split
profiles write the measured manylinux_2_35_x86_64; see SPLIT_PLAT). PyPI refuses that tag,
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

MERGED BINDINGS (2026-10-20, packaging/linux/merged_layout.py). Each set
directory (`<arch>/`, `<arch>/identical/`, `<arch>/host/`) ships one
`<name>.so` per binding, or the merged files of bindings/merged_groups.json
(`_mojolearn_native.so`, `_mojolearn_native_host.so`) plus the index
`_mojolearn_merged.json` naming each file's members. The layout is read from
the directory, one per wheel; the exactness checks hold the index members to
the same binding lists, the index ships beside its files, and
LINUX_PAYLOAD.json records it in `merged_indexes` with `binding_layout`.

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
release (`binding_origin`), and per set the same (`sets`). A file whose
record says `source: leg` was taken from a completed build leg of an earlier
freeze of the same release (tools/release.py: its set identity and build
tooling equal this freeze's); `from_legs` names that leg's commit, proof and
admission, and the set needs no proof here either.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from device_glue import DEVICE_GLUE, NO_IMAGE, validate as validate_device_glue
# THE BINDING LAYOUT OF A SET DIRECTORY (merged bindings, 2026-10-20): one
# <name>.so per binding, or one merged <stem>.so per group of
# bindings/merged_groups.json plus the index _mojolearn_merged.json that names
# each file's members. Read from each directory (index present = merged), never
# from MOJOLEARN_MERGED_BINDINGS; one layout per wheel. The index is shipped in
# every set directory beside the binaries, so it is in the wheel and its RECORD,
# and LINUX_PAYLOAD.json carries it in `merged_indexes` (the `.so` maps keep
# only binaries).
import merged_layout  # noqa: E402
MERGED_INDEX = merged_layout.INDEX
#: The binaries a host/ directory holds; the merged host file
#: `_mojolearn_native_host.so` matches it too.
HOST_GLOB = "_mojolearn_*_host.so"

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
    # The three TREE lanes: the only bindings with fast
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
#: CLASSICAL ML (2026-09-25): fast and identical.
FAST_CLASSICAL_NAMES = (
    "_mojolearn", "_mojolearn_estimators", "_mojolearn_svm", "_mojolearn_solver",
    "_mojolearn_metrics", "_mojolearn_preprocessing", "_mojolearn_tsa",
    "_mojolearn_linalg", "_mojolearn_arima", "_mojolearn_gp",
    "_mojolearn_kernel_methods", "_mojolearn_mixture", "_mojolearn_hdbscan",
    "_mojolearn_resample", "_mojolearn_ivf",
    # THE NEURAL BLOCKS (lane neural, 2026-09-27): fast and identical too.
    "_mojolearn_training", "_mojolearn_mamba",
    "_mojolearn_transformer", "_mojolearn_embedding",
)
#: Empty since the neural blocks gained FAST (lane neural, 2026-09-27); the
#: optional byte LM is added by `tier_names(include_byte_lm=True)`.
IDENTICAL_ONLY_NAMES = (
)
TIERS = ("fast", "identical")


def tier_names(tier, include_byte_lm=False, *, vendor=None):
    """Every extension expected in `tier`, in pack order.

    Trees exist in all tiers; classical/neural bindings in FAST and IDENTICAL.
    Optional ByteLM is IDENTICAL on Linux, FAST and IDENTICAL on Metal.
    A set on disk that does not match this exactly is refused in both directions.
    """
    names = EXT_NAMES
    if tier in ("identical", "fast"):
        names = names + FAST_CLASSICAL_NAMES
    if tier == "identical":
        names = names + IDENTICAL_ONLY_NAMES
    if include_byte_lm and (tier == "identical" or (tier == "fast" and vendor == "metal")):
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
#: `layout` is "merged" or "per-binding" (one for the fast, identical and host
#: directories of the set); `indexes` maps each merged directory's index,
#: set-relative ("_mojolearn_merged.json", "identical/...", "host/..."), to its
#: path (None or {} per-binding). `hosts` maps each host binding to the file it
#: loads from: its own file, or the merged host file for every member.
SetDir = collections.namedtuple("SetDir", "vendor arch files libs manifest hosts reuse layout indexes",
                                defaults=("per-binding", None))
# DEVIATION 2290. ONE reader for the release version and ONE spelling of the
# profile name, owned by tools/verify_linux_surface_qualification.py and shared
# with the qualification checker and the alpha artifact verifier. This packer
# used to carry its own regex over _version.py and the literals '0.6.1' and
# 'release-0.6.1'; the version was never published under that number.
sys.path.append(str(REPO / "tools"))
from verify_linux_surface_qualification import (  # noqa: E402
    OPTIONAL_ARCHES, PLUGIN_ARCHES, RELEASE_PROFILE, RELEASE_PROFILES, REGISTERED_ARCHES, load_gpu_plugins,
    normalise_arch, release_contract, release_version, wheel_host_bindings)
# THE SPLIT (2026-09-25): which plugin ships which vendor directory, read from
# python/mojolearn/gpu_plugins.py by path, the table the loader reads too.
gpu_plugins = load_gpu_plugins()
#: The profiles that emit the split wheels from the same inputs:
#: `mojolearn-core` (core-linux), `mojolearn-nvidia`, `mojolearn-nvidia-ptx`,
#: `mojolearn-amd` and the file-free `mojolearn` (meta-linux, requiring the
#: core and both vendors; 2026-10-20). `split` is the
#: default and checks sets the way `generic` does; `release-split` checks
#: them the way `release-linux3` does, per plugin (split_release_slots).
SPLIT_PROFILE = "split"
RELEASE_SPLIT_PROFILE = "release-split"
SPLIT_PROFILES = (SPLIT_PROFILE, RELEASE_SPLIT_PROFILE)
#: The wheels a split profile can emit; `--wheels` picks a subset.
WHEEL_KINDS = ((gpu_plugins.CORE_PROFILE,)
               + tuple(r["profile"] for r in gpu_plugins.distribution_rows(include_experimental=True))
               + (gpu_plugins.META_PROFILE,))
#: The profiles that carry no vendor: the core and the meta wheel.
NON_VENDOR_KINDS = (gpu_plugins.CORE_PROFILE, gpu_plugins.META_PROFILE)


def kind_distribution(kind):
    """The PyPI project of a split wheel kind (None is the core)."""
    if kind in (None, gpu_plugins.CORE_PROFILE):
        return gpu_plugins.CORE_DISTRIBUTION
    if kind == gpu_plugins.META_PROFILE:
        return gpu_plugins.META_DISTRIBUTION
    return gpu_plugins.package(kind)["distribution"]
#: The platform tag the split profiles write. MEASURED, not typed: auditwheel
#: measured the combined 0.8.16 and 0.8.18 wheels at manylinux_2_35_x86_64
#: (packaging/linux/audit.sh, show.txt), and every split wheel holds a subset
#: of those files, so none needs a newer glibc. audit.sh still re-measures
#: each one; the combined profiles keep writing linux_x86_64 for it to retag.
SPLIT_PLAT = "manylinux_2_35_x86_64"
# BY FILE PATH, not `from mojolearn import host_surface`: importing a
# submodule runs the package's __init__, which selects a backend and refuses
# on a box with no built binary (the release inventory test, Wheel CI).
# host_surface.py itself imports nothing from the package.
import importlib.util  # noqa: E402
_hs_spec = importlib.util.spec_from_file_location("mojolearn_host_surface", PY_DIR / "mojolearn" / "host_surface.py")
host_surface = importlib.util.module_from_spec(_hs_spec)
sys.modules[_hs_spec.name] = host_surface
_hs_spec.loader.exec_module(host_surface)
# THE EXPANSION LANES' GPU BINDINGS (lane/algos-prep, 2026-09-27), from the
# per-lane fragments the manifest merges; empty until a lane declares one.
# packaging/check_ext_lists.py holds this reader to the manifest.
FAST_CLASSICAL_NAMES += tuple(host_surface.expansion_gpu_bindings("classical"))
IDENTICAL_ONLY_NAMES += tuple(host_surface.expansion_gpu_bindings("identical-only"))


def split_release_slots(vendors):
    """The release architecture slots the plugins of `vendors` carry, for
    the split release profile, read from python/mojolearn/gpu_plugins.py
    (the table the loader reads): cuda -> sm_89 and the PTX slot sm_80,
    hip -> gfx942. The PTX slot is a set like any other: built by its own
    leg, proved by its own build proof (Andrew 2026-10-10)."""
    return {k for v in vendors for k in gpu_plugins.required_sets(v)}  # glue: union of registry slot sets


#: Every set a split release can be packed from (the core alone takes any of them).
RELEASE_SPLIT_SETS = {k for v in gpu_plugins.vendors() for k in gpu_plugins.required_sets(v)}  # glue: registry slot union
#: The sets the combined release-linux3 wheel must carry: the registry's
#: required sets of both vendors (gpu_plugins.required_sets; the admission side
#: reads the same, verify_linux_surface_qualification.RELEASE_ARCHES). The name
#: keeps the number the profile was authored under (DEVIATION 2290).
RELEASE_061_SETS = set(RELEASE_SPLIT_SETS)
#: Every set a release may carry: required plus registered-but-optional
#: (Hopper, Blackwell), as (vendor, arch).
RELEASE_REGISTERED_SETS = {tuple(k.split("/")) for k in REGISTERED_ARCHES}  # glue: registry key spelling


def release_inventory(sets, proof_paths, version, source_root=REPO, required=None,
                      profile=None):
    """Bind the explicit release-linux3 payload to complete per-architecture builds.

    File inspection only. Build provenance is not installed/runtime admission.
    Byte-LM is required only in IDENTICAL; legacy generic sets remain unchanged.
    `version` must be the version SOURCE_ROOT's _version.py declares
    (DEVIATION 2290); a literal never decides it.

    `required` is the set of architecture sets the sets must carry:
    RELEASE_061_SETS for the combined wheel, a package's share of it for the
    split profile, which packs a plugin from its own vendor's legs alone
    (NVIDIA and AMD release independently). A vendor whose required plugin
    sets are all in `required` may also carry its registered optional sets
    (Hopper, Blackwell; verify_linux_surface_qualification.OPTIONAL_ARCHES),
    each hardware slot once (sm_N or sm_Na, DEVIATION 2293). `profile` names
    the assembly profile recorded.
    """
    profile = profile or RELEASE_PROFILE
    required = RELEASE_061_SETS if required is None else set(required)
    keys = [(s.vendor, s.arch) for s in sets]
    keyset = set(keys)
    extra = keyset - required
    full_vendors = {v for v in PLUGIN_ARCHES if {tuple(k.split("/")) for k in PLUGIN_ARCHES[v]} <= required}  # glue: registry key spelling
    optional = {tuple(k.split("/")) for v in full_vendors for k in OPTIONAL_ARCHES[v]}  # glue: registry key spelling
    fills = (required <= keyset and extra <= optional
             and len({normalise_arch("/".join(k)) for k in keyset}) == len(keyset))  # glue: one set per hardware slot
    if version != read_version(source_root) or len(keys) != len(keyset) or not fills:
        raise SystemExit(profile + ' requires the sets '
                         + (release_contract(sorted({v for v, _ in required})) if required == RELEASE_061_SETS
                            else ', '.join('/'.join(k) for k in sorted(required)))
                         + ' (optional registered sets of a complete vendor allowed, each slot once); given '
                         + (', '.join('/'.join(k) for k in sorted(keys)) or 'none'))
    # ONE BINDING LAYOUT PER WHEEL (merged bindings): every set merged, or
    # every set per-binding; load_set already held each set to one.
    layouts = {s.layout or 'per-binding' for s in sets}
    if len(layouts) > 1:
        raise SystemExit(profile + ' requires one binding layout for every set; given '
                         + ', '.join(f'{s.vendor}/{s.arch}={s.layout}' for s in sets))
    binding_layout = next(iter(layouts), 'per-binding')
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
    # Per file it loads from (its own, or the merged host file and its index).
    host_built = {n for s in sets for n, p in (s.hosts or {}).items()
                  if f'host/{p.name}' not in reused_of[(s.vendor, s.arch)]
                  or (f'host/{MERGED_INDEX}' in (s.indexes or {})
                      and f'host/{MERGED_INDEX}' not in reused_of[(s.vendor, s.arch)])}
    proofs_needed = len(built_keys) if built_keys else (1 if host_built else 0)
    if len(proof_paths) != proofs_needed:
        raise SystemExit(profile + ' requires one complete build proof per set a leg built: '
                         f'{len(built_keys)} built ({", ".join("/".join(k) for k in sorted(built_keys)) or "none"}), '
                         f'{len(host_built)} host binding(s) built ({", ".join(sorted(host_built)) or "none"}), '
                         f'{proofs_needed} proof(s) needed, {len(proof_paths)} given')
    # Bindings only: the PTX set's PTX_BASELINE.json is its build manifest, bound
    # by the vendor marker, not a binding a build proof names.
    # `extensions` are the SHIPPED BINARIES: one per binding, or the merged files.
    payload = {f'mojolearn/{rel}': sha(path).hex()
               for s in sets for rel, path in s.files.items() if rel.endswith('.so')}
    # The indexes beside them, {'sha256', 'files'} each; the host index is one
    # copy for the wheel (mojolearn/host/), byte-identical across the sets.
    merged_indexes = {}
    for s in sets:
        for rel, path in sorted((s.indexes or {}).items()):
            name = f'mojolearn/{rel}' if rel.startswith('host/') else f'mojolearn/{s.vendor}/{s.arch}/{rel}'
            row = index_record(path)
            if merged_indexes.setdefault(name, row) != row:
                raise SystemExit(f'{profile}: the sets disagree on {name}; the host index is vendor-neutral, '
                                 'so one copy is wrong')
    origin = {}
    for s in sets:
        reused = reused_of[(s.vendor, s.arch)]
        for rel in (r for r in s.files if r.endswith('.so')):
            rec = reused.get(rel.split('/', 2)[2])
            origin[f'mojolearn/{rel}'] = (
                dict(origin='reused', set=f'{s.vendor}/{s.arch}', **reuse_origin(s, rec),
                     identity_digest=rec.get('identity_digest'), rebuilt_sha256=rec.get('rebuilt_sha256'))
                if rec else dict(origin='built', set=f'{s.vendor}/{s.arch}'))
    # The CPU training binding sits outside this map on purpose: every key here
    # is mojolearn/<vendor>/<arch>/..., and a vendor-neutral file belongs to no
    # architecture. It is recorded separately so the payload record still names
    # every shipped binary, with the legs that produced it and its digest.
    # Keyed by basename since 0.8.6: every host family the manifest ships,
    # each with the legs that built it and one digest (the byte compare in
    # main() has already refused legs that disagree).
    # Merged: every member's record names the merged host file it loads from.
    host_record = {}
    for name in HOST_NAMES:
        seen = {f'{s.vendor}/{s.arch}': sha(s.hosts[name]).hex() for s in sets if name in (s.hosts or {})}
        if seen:
            leaves = {s.hosts[name].name for s in sets if name in (s.hosts or {})}
            if len(leaves) != 1:
                raise SystemExit(f'{profile}: the sets load {name} from different files: {sorted(leaves)}')
            leaf = leaves.pop()
            taken = [s for s in sets if f'host/{leaf}' in reused_of[(s.vendor, s.arch)]]
            host_record[name] = dict(archive_path=f'mojolearn/host/{leaf}',
                                     sha256=sorted(set(seen.values()))[0],
                                     vendor='cpu', supported_modes=['identical'],
                                     unsupported_modes=['fast'],
                                     built_by=sorted(seen),
                                     origin='reused' if taken else 'built',
                                     **(reuse_origin(taken[0], taken[0].reuse['files'][f'host/{leaf}'])
                                        if taken else dict(from_release=None)),
                                     scope='CPU training and inference with no GPU; one '
                                           'copy for every architecture in this wheel')
            origin[f'mojolearn/host/{leaf}'] = dict(
                origin=host_record[name]['origin'], from_release=host_record[name]['from_release'],
                **({'from_leg': host_record[name]['from_leg']} if 'from_leg' in host_record[name] else {}),
                identity_digest=taken[0].reuse['files'][f'host/{leaf}'].get('identity_digest') if taken else None,
                built_by=[k for k in sorted(seen) if k not in {f'{s.vendor}/{s.arch}' for s in taken}])
    if set(host_record) != set(HOST_NAMES):
        raise SystemExit(profile + ' requires every host binding the manifest ships; missing: '
                         + ', '.join(sorted(set(HOST_NAMES) - set(host_record))))
    proofs, inventories, commits = {}, [], set()
    staged_keys = set()
    for path in proof_paths:
        raw = pathlib.Path(path).read_bytes()
        proof = json.loads(raw)
        from staged_native_admission import complete_native_proof
        if not complete_native_proof(proof, source_root):
            raise SystemExit('Incomplete architecture build or staged-admission proof')
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
        the_set = next(s for s in sets if (s.vendor, s.arch) == key)
        # Per-binding: one file per binding; merged: the planned files, their
        # members checked through the proof's merged_indexes below.
        if binding_layout == 'merged':
            required = {f'mojolearn/{key[0]}/{key[1]}/' + ('' if mode == 'fast' else mode + '/') + f
                        for mode in TIERS for f in merged_layout.plan(mode, True)}
        else:
            required = {f'mojolearn/{key[0]}/{key[1]}/' +
                        ('' if mode == 'fast' else mode + '/') + name + '.so'
                        for mode in TIERS for name in tier_names(mode, True)}
        if proof.get('binding_layout', 'per-binding') != binding_layout:
            raise SystemExit(f'The build proof of {"/".join(key)} records the binding layout '
                             f'{proof.get("binding_layout", "per-binding")!r}; the set is {binding_layout}')
        check_proof_indexes(proof, the_set, reused_of[key])
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
        from staged_native_admission import SCHEMA as STAGED_SCHEMA, decoded
        if proof['schema'] == STAGED_SCHEMA:
            staged_keys.add(key)
            stage = decoded(proof['stage_witness'])
            rows = {r['relative']: r for r in stage['outputs']}
            if 'canonical_host_stage_witness' in proof:
                rows.update({r['relative']:r for r in decoded(proof['canonical_host_stage_witness'])['outputs']
                             if r['relative'].startswith('host/')})
            for rel, row in rows.items():
                if merged_layout.is_index(rel):
                    continue  # no binary; its digest is in merged_indexes
                member = ('mojolearn/'+rel if rel.startswith('host/') else prefix+rel)
                origin[member].update(origin='staged-admission',
                    compiled_from_commit=row['original_stamp']['commit'],
                    unstaged_sha256=row['original_sha256'], source_closure=row['admitted_closure']['digest'])
                if rel.startswith('host/'):
                    # every binding the file serves (one per-binding, every member merged)
                    for rec in (r for r in host_record.values() if r['archive_path'] == member):
                        rec['origin'] = 'staged-admission'
                        rec['admitted_by'] = rec.pop('built_by', rec.get('admitted_by', []))

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
            set_records['/'.join(key)] = dict(proofs[key], origin=('staged-admission' if key in staged_keys else 'built') if not reused_of[key] else 'mixed',
                                              reused=sorted(reused_of[key]))
        else:
            set_records['/'.join(key)] = dict(origin='reused', from_release=s.reuse['origin'],
                                              reused=sorted(reused_of[key]),
                                              **({'from_legs': s.reuse['legs']} if s.reuse.get('legs') else {}))
    n_reused = sum(1 for o in origin.values() if o['origin'] == 'reused')
    n_admitted = sum(1 for o in origin.values() if o['origin'] == 'staged-admission')
    return dict(schema='mojolearn.linux-payload.v1', version=version,
                release_profile='alpha-api', assembly_profile=profile,
                source_commit=commit, source_inventory=inventory,
                sets=set_records,
                extensions=payload,
                merged_indexes=merged_indexes,
                binding_layout=binding_layout,
                binding_origin=origin,
                reuse=dict(built=len(origin) - n_reused - n_admitted, reused=n_reused,
                           **({'staged_admitted': n_admitted} if n_admitted else {}),
                           from_release=next((s.reuse['origin'] for s in sets if s.reuse and s.reuse['origin']), None),
                           from_legs={k: v for s in sets if s.reuse for k, v in (s.reuse.get('legs') or {}).items()}),
                optional_native={**{n: {
                    'included': True, 'supported_modes': ['identical'],
                    'unsupported_modes': ['fast']}
                    for n in ('_mojolearn_byte_lm',) + tuple(host_record) + IDENTICAL_ONLY_NAMES},
                    **{n: {
                    'included': True, 'supported_modes': ['fast', 'identical'],
                    'unsupported_modes': []}
                    for n in FAST_CLASSICAL_NAMES}},
                # One record per host binding, keyed by basename, so the
                # payload says which host families this wheel carries rather
                # than leaving a reader to infer it from the archive.
                host_native=host_record,
                qualification='Build and file provenance only; installed runtime and numerical checks required',
                runtime_coverage={'/'.join(k): 'PENDING_INSTALLED_ARTIFACT' for k in sorted(keys)})


def check_proof_indexes(proof, s, reused):
    """A build proof's `merged_indexes` against set S (one SetDir).

    Merged: one row per merged directory of the set (fast, identical/, and
    host/ when the set carries host bindings), keyed
    mojolearn/<vendor>/<arch>/[identical/|host/]_mojolearn_merged.json; each
    row's members are exactly that directory's bindings (the byte LM once, in
    the identical index alone) and its sha256 is the shipped index's, unless
    the set took that index from the published wheel. Per-binding: none."""
    got = proof.get('merged_indexes') or {}
    prefix = f'mojolearn/{s.vendor}/{s.arch}/'
    if (s.layout or 'per-binding') != 'merged':
        if got:
            raise SystemExit(f'The build proof of {s.vendor}/{s.arch} carries merged indexes; the set is per-binding')
        return
    want = {sub + MERGED_INDEX: (set(tier_names(tier, True)) if tier != 'host' else set(HOST_NAMES))
            for tier, sub in (('fast', ''), ('identical', 'identical/'), ('host', 'host/'))
            if sub + MERGED_INDEX in (s.indexes or {})}
    if not isinstance(got, dict) or set(got) != {prefix + rel for rel in want}:
        raise SystemExit(f'The build proof of {s.vendor}/{s.arch} names merged indexes '
                         f'{sorted(got) if isinstance(got, dict) else got!r}; the set ships '
                         f'{sorted(prefix + rel for rel in want)}')
    for rel, names in want.items():
        row = got[prefix + rel]
        files = row.get('files') if isinstance(row, dict) else None
        members = [n for m in (files or {}).values() for n in (m if isinstance(m, list) else [None])]
        if not isinstance(files, dict) or len(members) != len(set(members)) or set(members) != names:
            raise SystemExit(f'The build proof of {s.vendor}/{s.arch}: {rel} does not index exactly '
                             'the bindings of its directory, each once')
        if rel not in reused and row.get('sha256') != sha(s.indexes[rel]).hex():
            raise SystemExit(f'Duplicate, stale or incomplete architecture proof: {rel} differs from the shipped index')


def urlsafe_b64(digest):
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def read_version(root=REPO):
    """`__version__` of ROOT/python/mojolearn/_version.py through the shared reader (DEVIATION 2290)."""
    try:
        return release_version(root)
    except (OSError, ValueError) as exc:
        raise SystemExit("pack_wheel: " + str(exc))


def metadata_text(proj, readme):
    """Metadata 2.4, field order as setuptools 84 wrote it for 0.1.0.

    The split core's METADATA is the combined wheel's under the name
    mojolearn-core, with no vendor requirement (core_project); verification
    stays optional. Each plugin pins the core (plugin_project); the Linux
    mojolearn pins the core and both vendors (meta_project)."""
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
    for extra, dependencies in proj.get("optional-dependencies", {}).items():
        lines.append(f"Provides-Extra: {extra}")
        for dependency in dependencies:
            requirement, _, marker = dependency.partition(";")
            condition = f'({marker.strip()}) and ' if marker else ''
            lines.append(f'Requires-Dist: {requirement.strip()}; {condition}extra == "{extra}"')
    for d in proj.get("dependencies", []):
        lines.append(f"Requires-Dist: {d}")
    lines.append("Dynamic: license-file")
    return "\n".join(lines) + "\n\n" + readme


#: The note the core's README opens with (user voice, short).
CORE_README_NOTE = (
    "# mojolearn-core\n\n"
    "The mojolearn Python package and CPU runtime for Linux x86-64, with no GPU binaries. "
    "`pip install mojolearn` installs it with NVIDIA and AMD support; "
    "`pip install mojolearn-nvidia` or `pip install mojolearn-amd` installs it with one vendor. "
    "There is no macOS wheel of mojolearn-core: on a Mac, `pip install mojolearn`.\n\n")


def core_project(proj, version):
    """The split core's pyproject-shaped fields: the combined wheel's under
    the name mojolearn-core, with no vendor requirement
    (gpu_plugins.core_requirements is empty since 2026-10-20: each vendor
    requires the core, and `mojolearn` requires all of them)."""
    out = dict(proj)
    out["name"] = gpu_plugins.CORE_DISTRIBUTION
    out["description"] = ("mojolearn for Linux x86-64 without GPU binaries; "
                          "add mojolearn-nvidia or mojolearn-amd, or pip install mojolearn for both")
    out["dependencies"] = list(proj.get("dependencies", [])) + gpu_plugins.core_requirements(version)
    return out


def meta_project(proj, version):
    """The Linux `mojolearn` wheel's fields: no files, no extras, and exactly
    the core and both vendors at this version (gpu_plugins.meta_requirements)."""
    out = {k: proj[k] for k in ("name", "authors", "maintainers", "license", "urls", "keywords",
                                "classifiers", "requires-python", "license-files", "description") if k in proj}
    out.update(name=gpu_plugins.META_DISTRIBUTION, version=version,
               dependencies=gpu_plugins.meta_requirements(version))
    readme = (f"# mojolearn\n\n"
              f"On Linux x86-64, `pip install mojolearn` installs mojolearn-core with NVIDIA "
              f"(mojolearn-nvidia) and AMD (mojolearn-amd) support. "
              f"To install one vendor only: `pip install mojolearn-nvidia` or `pip install mojolearn-amd`.\n")
    return out, readme


def plugin_project(proj, vendor, version, arches):
    """Project metadata for a vendor or experimental payload profile."""
    row = gpu_plugins.package(vendor)
    out = {k: proj[k] for k in ("authors", "maintainers", "license", "urls", "keywords",
                                "classifiers", "requires-python", "license-files") if k in proj}
    out.update(name=row["distribution"], version=version,
               description=(f"{row['label']} GPU binaries for mojolearn {version} "
                            f"({', '.join(sorted(arches))}); installs mojolearn-core with them"),
               dependencies=gpu_plugins.package_requirements(vendor, version))
    readme = (f"# {row['distribution']}\n\n"
              f"{row['label']} {row['role']} for mojolearn {version}. "
              f"`pip install {row['distribution']}` installs mojolearn-core and this package; "
              "`pip install mojolearn` installs both vendor packages and all released native targets. "
              "Each vendor wheel contains its native architecture sets; the loader selects the matching set.\n")
    # tools/release.py sets this when a native set of the release had no installed smoke (2026-10-08: releases rent
    # nothing, so sm_90a is smoked only when a Hopper box is held); the wheel says so instead of implying a test
    unsmoked = sorted(a for a in os.environ.get("MOJOLEARN_RELEASE_UNSMOKED_ARCHES", "").split(",") if a in arches)
    if unsmoked:
        readme += (f"\nInstalled smoke: not run for {', '.join(unsmoked)} in this release (no box of that architecture "
                   "was held); those sets ship on their build receipt. The other sets passed the installed smoke.\n")
    return out, readme


def wheel_file_text(tag):
    return ("Wheel-Version: 1.0\nGenerator: mojolearn pack_wheel.py\n"
            f"Root-Is-Purelib: false\nTag: {tag}\n")


def write_wheel(whl, entries, generated, dist):
    """Write one wheel: `entries` (archive path -> file) then `generated`
    (archive path -> bytes), in that order, and a RECORD. The combined
    wheel's writer, unchanged, shared by the split profiles."""
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
    return whl


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


def _layout_error(exc):
    return SystemExit(f"pack_wheel: undeclared or missing native payload: {exc}")


def set_dir_layout(directory, names, glob="_mojolearn*.so", plan_key=None, byte_lm=False):
    """("merged" | "per-binding", {file.so: [members]}) of one set directory
    that must serve exactly `names` (merged_layout.layout). A merged
    directory's index must also be the bindings/merged_groups.json plan for
    `plan_key` (fast, identical, host). SystemExit otherwise."""
    try:
        kind, files = merged_layout.layout(directory, names, glob)
        if kind == "merged" and plan_key:
            merged_layout.check_plan(directory, plan_key, byte_lm)
    except merged_layout.LayoutError as exc:
        raise _layout_error(exc) from exc
    return kind, files


def read_set_index(directory):
    """The directory's merged index {file.so: [members]}, or None (per-binding)."""
    try:
        return merged_layout.read_index(directory)
    except merged_layout.LayoutError as exc:
        raise _layout_error(exc) from exc


def binding_file(directory, name):
    """The file binding `name` loads from in `directory`: its own file, else
    the merged file the index names (the loader's rule)."""
    try:
        return pathlib.Path(merged_layout.member_file(directory, name))
    except merged_layout.LayoutError as exc:
        raise _layout_error(exc) from exc


def index_record(path):
    """{'sha256': hex, 'files': {file.so: [members]}} of one shipped index (a
    `merged_indexes` row, the shape tools/verify_linux_surface_qualification.py reads)."""
    files = read_set_index(pathlib.Path(path).parent)
    return dict(sha256=sha(path).hex(), files={f: list(m) for f, m in sorted(files.items())})


def host_witnesses(set_paths):
    """name -> sha256 of every host binding some architecture directory under
    `set_paths` READ BACK as `cpu` (a `host <name> cpu` row in its
    readback.txt, the binary it loads from beside it: its own file, or the
    merged host file). A host binding has no device code and
    the same bytes are placed into every set; a set that reuses every GPU
    binding from the published wheel has no read-back of its own, so a host
    binding built for this release is proven there by the leg that built and
    read it back, byte for byte (release_reuse's linux-wait requires the host
    bindings to be byte-identical across the legs). A merged host directory's
    index is witnessed the same way, under the key MERGED_INDEX.

    The layout check comes first: host files of a merged and a per-binding set
    differ by construction, so a mix is refused as a mix (the same message as
    main's per-set check), never as a byte mismatch."""
    layouts = {}
    for sp in set_paths:
        sp = pathlib.Path(sp).resolve()
        for adir in sorted(d for d in sp.iterdir() if d.is_dir() and ARCH_RE.match(d.name)):
            if (adir / "readback.txt").is_file() and (adir / "host").is_dir():
                layouts[f"{sp.name}/{adir.name}"] = ("merged" if (adir / "host" / MERGED_INDEX).is_file()
                                                     else "per-binding")
    if len(set(layouts.values())) > 1:
        raise SystemExit("pack_wheel: the sets mix the merged and per-binding layouts: "
                         + ", ".join(f"{k}={v}" for k, v in sorted(layouts.items())))
    out = {}
    for sp in set_paths:
        sp = pathlib.Path(sp).resolve()
        for adir in sorted(d for d in sp.iterdir() if d.is_dir() and ARCH_RE.match(d.name)):
            rb = adir / "readback.txt"
            if not rb.is_file():
                continue
            witnessed = False
            for line in rb.read_text().splitlines():
                row = line.split()
                if len(row) != 3 or row[0] != "host" or row[2] != "cpu":
                    continue
                so = binding_file(adir / "host", row[1])
                if so.is_file():
                    witnessed = True
                    digest = sha(so)
                    if out.setdefault(row[1], digest) != digest:
                        raise SystemExit(f"pack_wheel: host binding {row[1]} was read back with different bytes in two sets; "
                                         "one of them is not this release's")
            index = adir / "host" / MERGED_INDEX
            if witnessed and index.is_file():
                digest = sha(index)
                if out.setdefault(MERGED_INDEX, digest) != digest:
                    raise SystemExit("pack_wheel: the merged host index was read back with different bytes in two "
                                     "sets; one of them is not this release's")
    return out


def load_set(path, include_byte_lm=False, host_witnesses_by_name=None):
    """Every (vendor, arch, files, libs, manifest) under one sets/<vendor>
    directory. One tuple per architecture subdirectory. `host_witnesses_by_name`
    (see `host_witnesses`) lets a set that reuses every GPU binding carry a
    host binding built for this release when a witnessed set holds the same
    bytes. Each directory of a set (fast, identical/, host/) is per-binding or
    merged (merged_layout); a set never mixes the two."""
    path = pathlib.Path(path).resolve()
    vendor = path.name
    if vendor not in LINUX_VENDORS:
        raise SystemExit(f"pack_wheel: set directory must be named cuda or hip: {path}")
    if any(path.glob("_mojolearn*.so")) or (path / MERGED_INDEX).exists():
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
        # Experimental baselines use the generic profile, but a full baseline
        # build can include byte LM. Preserve that declared file and enforce
        # complete readbacks when present. Native generic semantics are unchanged.
        # Merged: the byte LM is a member of the identical index.
        carry_byte_lm = include_byte_lm or (
            vendor == "cuda" and arch == "sm_80"
            and (adir / gpu_plugins.BASELINE_MANIFEST).is_file()
            and binding_file(adir / "identical", "_mojolearn_byte_lm").is_file())
        manifest = json.loads((adir / "manifest.json").read_text())
        reuse = load_reuse(adir, vendor, arch)
        reused = reuse["files"] if reuse else {}
        # EVERY TIER DIRECTORY'S LAYOUT FIRST, exact both ways: per-binding,
        # one <name>.so per expected binding; merged, the planned files whose
        # index members are exactly the expected bindings (the byte LM in the
        # identical index once and in no fast index), no own member file
        # beside the index.
        tier_layouts = {}
        for tier in TIERS:
            d = adir if tier == "fast" else adir / tier
            tier_layouts[tier] = set_dir_layout(d, tier_names(tier, carry_byte_lm),
                                                plan_key=tier, byte_lm=carry_byte_lm)
        host_dir = adir / "host"
        host_index = read_set_index(host_dir)

        def host_rels(names):
            # set-relative files host/ ships for `names`: their own files, or
            # the merged files serving them and the index
            if host_index is None:
                return {f"host/{n}.so" for n in names}
            return ({f"host/{f}" for f, m in host_index.items() if set(m) & set(names)}
                    | {f"host/{MERGED_INDEX}"})

        expected_rels = set()
        for tier, (kind, tfiles) in tier_layouts.items():
            sub = "" if tier == "fast" else tier + "/"
            expected_rels |= {sub + f for f in tfiles} | ({sub + MERGED_INDEX} if kind == "merged" else set())
        all_reused = bool(reuse) and expected_rels <= set(reused) and (
            not carry_byte_lm or host_rels(HOST_NAMES) <= set(reused))
        has_witnesses = (adir / "readback.txt").is_file() and (adir / "arch_readback.txt").is_file()
        witnessed_built = []
        if not has_witnesses and not all_reused:
            missing = expected_rels - set(reused)
            if carry_byte_lm:
                missing |= host_rels(HOST_NAMES) - set(reused)
            built_host = {rel for rel in missing if rel.startswith("host/")}
            if not reuse or built_host != missing or not host_witnesses_by_name:
                raise SystemExit(
                    f"pack_wheel: {adir} has no readback.txt or arch_readback.txt and not every binding "
                    "in it is taken from the published wheel (reuse.json); a built binding needs its read-back")
            # EVERY GPU BINDING IS THE PUBLISHED ONE; the host bindings built
            # for this release were read back as `cpu` by the leg that built
            # them, in a set that holds the same bytes (a merged host file:
            # every member it serves, and the index beside it).
            for rel in sorted(built_host):
                leaf = rel[len("host/"):]
                so = adir / rel
                if leaf == MERGED_INDEX:
                    served = []
                    ok = so.is_file() and host_witnesses_by_name.get(MERGED_INDEX) == sha(so)
                else:
                    served = list(host_index[leaf]) if host_index is not None else [leaf[:-len(".so")]]
                    ok = so.is_file() and all(host_witnesses_by_name.get(n) == sha(so) for n in served)
                if not ok:
                    raise SystemExit(
                        f"pack_wheel: {adir}/{rel} was built for this release and no set with a read-back holds "
                        "the same bytes; a built binding needs its read-back")
                witnessed_built.extend(served)
        if not has_witnesses:
            # EVERY GPU BINDING WAS TAKEN FROM THE PUBLISHED WHEEL. Its vendor and
            # architecture were read back on the box that built it, for the
            # release that shipped it; the bytes verified above are those. A
            # host binding built for this release is witnessed in another set.
            host_named = [n for n in HOST_NAMES
                          if "host/" + binding_file(host_dir, n).name in reused or n in witnessed_built]
        else:
            # THE HOST ROWS ARE READ SEPARATELY, and by name. The CPU training
            # binding has no device code and answers 'cpu', so folding it into the
            # vendor and architecture comparisons below would either refuse a
            # correct set or force those comparisons to accept 'cpu' from a GPU
            # binary, which is the failure they exist to catch. Rows whose first
            # field is `host` are pulled out here and checked on their own terms.
            # Rows name each binding in either layout (a merged file's members
            # are read back one by one).
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
            if carry_byte_lm and set(host_named) != set(HOST_NAMES):
                raise SystemExit(
                    f"pack_wheel: {adir}/readback.txt names {sorted(host_named)}; the release profile "
                    f"requires every host binding the manifest ships: {list(HOST_NAMES)}")
            # THE ARCHITECTURE IS VERIFIED THE SAME WAY THE VENDOR IS: read back
            # from the binaries on the box (build_sets.sh), never typed. A set
            # whose read-back disagrees with its directory name is refused, the
            # exact failure mode that shipped 0.3.0 as sm_90a-only.
            ab_lines = [ln.split() for ln in (adir / "arch_readback.txt").read_text().splitlines()]
            try:
                validate_device_glue(ab_lines, adir)
            except (ValueError, OSError) as exc:
                raise SystemExit(f"pack_wheel: {exc}") from exc
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
            if carry_byte_lm:
                expected_rows = {(tier, name) for tier in TIERS
                                 for name in tier_names(tier, True)}
                for witness, expected_value in (('readback.txt', vendor), ('arch_readback.txt', arch)):
                    rows = [line.split() for line in (adir / witness).read_text().splitlines()
                            if not line.startswith('host ')]
                    if (len(rows) != len(expected_rows) or any(len(row) != 3 for row in rows)
                            or {(row[0], row[1]) for row in rows} != expected_rows
                            or any(row[2] != expected_value and not (
                                witness == "arch_readback.txt" and row[2] == NO_IMAGE
                                and (row[0], row[1]) in DEVICE_GLUE) for row in rows)):
                        raise SystemExit(f'pack_wheel: incomplete release native readback in {adir / witness}')
        # OPTIONAL WHEN ABSENT (generic profile only), because a set built
        # before this payload existed is still a valid set and a rebuild of an
        # older commit must not be refused. A named binding must be on disk;
        # a file on disk that no row names is refused, because an unread
        # binary is one whose vendor and column nobody checked. `host_payload`
        # is a dict basename -> the file it loads from (its own, or the merged
        # host file), empty when the set carries none, and `main` says so once
        # rather than shipping an export that cannot work.
        indexes = {}
        host_payload = {}
        if host_index is None:
            for name in host_named:
                host_so = host_dir / f"{name}.so"
                if not host_so.exists():
                    raise SystemExit(
                        f"pack_wheel: {adir}/readback.txt names {name} but {host_so} is absent")
                host_payload[name] = host_so
            on_disk = {p.name[:-3] for p in host_dir.glob(HOST_GLOB)} if host_dir.is_dir() else set()
            if on_disk - set(host_named):
                raise SystemExit(
                    f"pack_wheel: {adir}/host/ holds bindings readback.txt never read back: "
                    f"{sorted(on_disk - set(host_named))}")
            host_layout = "per-binding" if host_named else None
        else:
            # MERGED HOST: the index's members are exactly the bindings read
            # back (an indexed member nobody read back is an unread binary),
            # and the index is the manifest's host plan.
            _, hfiles = set_dir_layout(host_dir, host_named, HOST_GLOB, plan_key="host", byte_lm=carry_byte_lm)
            host_payload = {n: host_dir / f for f, m in hfiles.items() for n in m}
            indexes[f"host/{MERGED_INDEX}"] = host_dir / MERGED_INDEX
            host_layout = "merged"
        kinds = {kind for kind, _ in tier_layouts.values()} | ({host_layout} if host_layout else set())
        if len(kinds) != 1:
            raise SystemExit(f"pack_wheel: {adir} mixes the merged and per-binding layouts across its "
                             "fast, identical and host directories; a set is built in one layout")
        layout = next(iter(kinds))
        files = {}
        for tier in TIERS:
            d = adir if tier == "fast" else adir / tier
            kind, tfiles = tier_layouts[tier]
            sub = "" if tier == "fast" else tier + "/"
            for f in sorted(tfiles):
                so = d / f
                if not so.exists():
                    raise SystemExit(
                        f"pack_wheel: {vendor}/{arch} {tier} set is missing {f}")
                files[f"{vendor}/{arch}/{sub}{f}"] = so
            if kind == "merged":
                # the index ships beside its files: the loader trusts it
                files[f"{vendor}/{arch}/{sub}{MERGED_INDEX}"] = d / MERGED_INDEX
                indexes[sub + MERGED_INDEX] = d / MERGED_INDEX
        baseline_path = adir / gpu_plugins.BASELINE_MANIFEST
        if vendor == "cuda" and arch == gpu_plugins.PTX_ARCH:
            if not baseline_path.is_file():
                raise SystemExit("pack_wheel: the PTX set (cuda/sm_80) requires its PTX_BASELINE.json")
            doc = json.loads(baseline_path.read_text())
            # binaries only: an index is no PTX row; a merged set's manifest binds
            # its tier indexes in `indexes` ({} per-binding, which must list none)
            digests = {rel.split("/", 2)[2]: sha(src).hex() for rel, src in files.items() if rel.endswith(".so")}
            tier_indexes = {rel: sha(p).hex() for rel, p in indexes.items() if not rel.startswith("host/")}
            try:
                gpu_plugins.validate_baseline_manifest(doc, digests, indexes=tier_indexes)
            except ValueError as exc:
                raise SystemExit(f"pack_wheel: invalid PTX set manifest: {exc}") from exc
            files[f"{vendor}/{arch}/{gpu_plugins.BASELINE_MANIFEST}"] = baseline_path
        elif baseline_path.exists():
            raise SystemExit("pack_wheel: PTX manifest on a native architecture set")
        libs = {p.name: p for p in sorted((adir / ".libs").glob("*"))}
        if not libs:
            raise SystemExit(f"pack_wheel: {adir}/.libs is empty; stage_libs.py did not run")
        if reuse:
            # `files` is keyed vendor/arch/<set-relative>; reuse.json is set-relative,
            # one record per shipped file in either layout (an index included)
            packed = {rel.split("/", 2)[2] for rel in files}
            host_shipped = {f"host/{p.name}" for p in host_payload.values()} | {
                rel for rel in indexes if rel.startswith("host/")}
            unknown = set(reused) - packed - host_shipped - {f".libs/{n}" for n in libs}
            if unknown:
                raise SystemExit(f"pack_wheel: {adir}/reuse.json names files this set does not pack: {sorted(unknown)}")
        out.append(SetDir(vendor, arch, files, libs, manifest, host_payload, reuse, layout, indexes))
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
    legs = doc.get("from_legs") or {}
    files = doc.get("files")
    # A file comes from the published wheel (`source` release, the default)
    # or from a completed build leg of an earlier freeze of this release whose
    # set identity equals this freeze's (`source` leg, tools/release.py).
    sources = {(rec.get("source") or "release") if isinstance(rec, dict) else None
               for rec in (files or {}).values()} if isinstance(files, dict) else {None}
    published_ok = all(origin.get(k) for k in ("version", "source_commit", "wheel", "wheel_sha256"))
    legs_ok = isinstance(legs, dict) and all(
        isinstance(o, dict) and all(o.get(k) for k in LEG_ORIGIN_KEYS) for o in legs.values())
    if (doc.get("schema") != REUSE_SCHEMA or doc.get("set") != f"{vendor}/{arch}"
            or not isinstance(files, dict) or not sources <= {"release", "leg"}
            or (("release" in sources or not files) and not published_ok)
            or ("leg" in sources and not legs_ok)
            or any(rec.get("leg") not in legs for rec in files.values() if rec.get("source") == "leg")):
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
            whence = (f"leg {rec.get('leg')} of {legs[rec['leg']]['source_commit'][:12]}"
                      if (rec.get("source") or "release") == "leg" else f"published {origin['version']}")
            raise SystemExit(
                f"pack_wheel: {adir}/{rel} differs from the {whence} bytes reuse.json "
                f"records ({want[:12]}); a reused binding must be those bytes, refusing to pack it")
    return dict(files=doc["files"],
                origin=dict(version=origin["version"], source_commit=origin["source_commit"],
                            wheel=origin["wheel"], wheel_sha256=origin["wheel_sha256"]) if published_ok else None,
                legs={name: dict(o) for name, o in legs.items()})


#: What a reused build leg's origin must name (tools/release.py writes it).
LEG_ORIGIN_KEYS = ("source_commit", "leg", "proof_sha256", "admitted_for", "set_identity_digest", "tooling_digest")


def reuse_origin(s, rec):
    """The origin fields of one reused file of set S: the published release
    it came from, or the earlier freeze's build leg."""
    if (rec.get("source") or "release") == "leg":
        return dict(from_release=None, from_leg=s.reuse["legs"][rec["leg"]])
    return dict(from_release=s.reuse["origin"])


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


def main(argv=None, _gates=True):
    """Pack. Returns 0, or 1 when a wheel is over PyPI's limit.

    `_gates=False` is for the file-only tests alone (packaging/linux/
    test_split_wheels.py), whose sets are inert bytes rather than ELF: it
    skips the two steps that read binaries and the checkout's API surface
    (portable_math/wheel.py and wheel_api_audit.audit). No command line
    reaches it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True,
                    help="a sets/<vendor> directory from build_sets.sh; give both")
    ap.add_argument("--out", default=str(PY_DIR / "dist"))
    ap.add_argument("--plat", default="",
                    help="platform tag; default " + SPLIT_PLAT + " for the split profiles, "
                         "linux_x86_64 (for audit.sh to retag) for the combined ones")
    # DEVIATION 2290: `release-0.6.1` is the deprecated alias of release-linux3.
    # THE DEFAULT IS THE SPLIT: the core, the PTX payload and one
    # plugin per GPU vendor. `generic` and `release-linux3` still pack the one
    # combined wheel and are what tools/release.py asks for by name.
    ap.add_argument('--profile', choices=SPLIT_PROFILES + ('generic', RELEASE_PROFILE, 'release-0.6.1'),
                    default=SPLIT_PROFILE)
    ap.add_argument('--wheels', default='',
                    help='split profiles: comma list of ' + ','.join(WHEEL_KINDS)
                         + '; default the core plus the plugin of every vendor given')
    ap.add_argument('--build-proof', action='append', default=[],
                    help='complete per-architecture build-provenance.json; one per built set for '
                         + RELEASE_PROFILE + ' and ' + RELEASE_SPLIT_PROFILE)
    ap.add_argument("--check-against", default="",
                    help="a macOS wheel whose METADATA must match this one's")
    ap.add_argument("--portable-math-helper", type=pathlib.Path,
                    help="Linux-built libMojolearnMath.so; required when packing on macOS")
    a = ap.parse_args(argv)
    if a.profile in RELEASE_PROFILES:
        a.profile = RELEASE_PROFILE  # DEVIATION 2290: the alias maps to the same path
    split = a.profile in SPLIT_PROFILES
    strict = a.profile in (RELEASE_PROFILE, RELEASE_SPLIT_PROFILE)
    if a.wheels and not split:
        raise SystemExit("pack_wheel: --wheels needs a split profile (split, release-split)")
    plat = a.plat or (SPLIT_PLAT if split else "linux_x86_64")

    proj = tomllib.loads((PY_DIR / "pyproject.toml").read_text())["project"]
    version = read_version()
    if proj["version"] != version:
        raise SystemExit(f"pack_wheel: pyproject says {proj['version']}, "
                         f"_version.py says {version}")
    readme = (REPO / "README.md").read_text()

    witnesses = host_witnesses(a.set)
    sets = [t for s in a.set for t in load_set(s, include_byte_lm=strict, host_witnesses_by_name=witnesses)]
    keys = [(s.vendor, s.arch) for s in sets]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"pack_wheel: the same (vendor, arch) given twice: {keys}")
    # ONE BINDING LAYOUT PER WHEEL: the loader reads each directory's own
    # layout, but a wheel whose sets differ is two builds packed as one.
    if len({s.layout for s in sets}) > 1:
        raise SystemExit("pack_wheel: the sets mix the merged and per-binding layouts: "
                         + ", ".join(f"{s.vendor}/{s.arch}={s.layout}" for s in sets))
    print(f"pack_wheel: binding layout {sets[0].layout}")
    if not strict and a.build_proof:
        raise SystemExit('--build-proof requires an explicit release profile')
    if not strict and any(s.reuse for s in sets):
        raise SystemExit('pack_wheel: reuse.json (bindings taken from a published wheel) needs the release profile')
    kinds = split_kinds(a.wheels, {v for v, _ in keys}, keys) if split else ()
    # THE PTX SLOT (Andrew 2026-10-10: PTX is a normal target; no flag). The
    # cuda/sm_80 set is the PTX set; it packs into mojolearn-nvidia beside the
    # native sets, and the vendor marker binds its manifest by SHA256.
    bundle = None
    ptx_key = ("cuda", gpu_plugins.PTX_ARCH)
    if ptx_key in keys:
        if not split:
            raise SystemExit("pack_wheel: the PTX slot (cuda/sm_80) packs only into the split "
                             "mojolearn-nvidia wheel (--profile split or release-split)")
        ptx_set = next(s for s in sets if (s.vendor, s.arch) == ptx_key)
        manifest_path = ptx_set.files[f"cuda/{gpu_plugins.PTX_ARCH}/" + gpu_plugins.BASELINE_MANIFEST]
        manifest_bytes = manifest_path.read_bytes()
        bundle = dict(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
        try:
            gpu_plugins.validate_bundled_ptx(bundle, manifest_bytes,
                {rel.split("/", 2)[2]: sha(path).hex() for rel, path in ptx_set.files.items() if rel.endswith(".so")},
                indexes={rel.split("/", 2)[2]: sha(path).hex() for rel, path in ptx_set.files.items()
                         if merged_layout.is_index(rel)})
        except ValueError as exc:
            raise SystemExit(f"pack_wheel: invalid PTX set: {exc}") from exc
    if a.profile == RELEASE_PROFILE:
        inventory = release_inventory(sets, a.build_proof, version)
    elif a.profile == RELEASE_SPLIT_PROFILE:
        plugin_vendors = {gpu_plugins.by_profile(k) for k in kinds if k not in NON_VENDOR_KINDS}
        # A plugin is packed from its own vendor's sets and nothing else; the
        # core alone (host bindings and runtime) from whichever release sets
        # were given. Either way every set given is proven like release-linux3's,
        # the PTX set by its own leg's proof.
        package_slots = set()
        for kind in kinds:
            if kind in NON_VENDOR_KINDS:
                continue
            package = gpu_plugins.package(kind)
            if package["role"] == "vendor":
                package_slots.update(split_release_slots({package["vendor"]}))
            else:
                package_slots.update((package["vendor"], slot[0]) for slot in package["slots"])
        required = (package_slots if plugin_vendors
                    else set(keys) & RELEASE_REGISTERED_SETS)
        inventory = release_inventory(sets, a.build_proof, version, required=required,
                                      profile=RELEASE_SPLIT_PROFILE)
    else:
        inventory = None
    if inventory is not None:
        if bundle and json.loads(manifest_bytes)["source_commit"] != inventory["source_commit"]:
            raise SystemExit("pack_wheel: the PTX set and the native sets were built from different commits")
        if split:
            for mapping in ("extensions", "binding_origin", "merged_indexes"):
                inventory[mapping] = {gpu_plugins.installed_member(k): v
                                      for k, v in inventory[mapping].items()}
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
    if split and not shared:
        # THE CORE CARRIES THE RUNTIME its host bindings load, at
        # mojolearn/.libs, where every set's RUNPATH also looks. A per-vendor
        # closure would have to ship inside a plugin and leave the core's host
        # bindings with none, so the split refuses rather than lay that out.
        raise SystemExit(
            "pack_wheel: the split profiles need ONE runtime closure shared by every set "
            "(mojolearn/.libs in the core); these sets' closures differ: "
            + ", ".join(f"{v}/{arch}" for v, arch in keys))
    entries = {}  # archive path -> filesystem path
    entries["mojolearn_diagnostics.py"] = PY_DIR / "mojolearn_diagnostics.py"
    entries.update(python_package_entries())
    entries["mojolearn/ALPHA_API.md"] = PKG / "ALPHA_API.md"
    entries["mojolearn/models/catalog.json"] = PKG / "models" / "catalog.json"  # mojolearn.models.catalog()
    seen_vendor_libs = set()
    for s in sets:
        vendor, files, libs = s.vendor, s.files, s.libs
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
        leaves = {p.name for p in host_payloads.values()}
        if len(leaves) != 1:
            raise SystemExit(f"pack_wheel: the architecture legs load {name} from different files: {sorted(leaves)}")
        # its own file, or the merged host file (one entry for all its members)
        entries[f"mojolearn/host/{leaves.pop()}"] = next(iter(host_payloads.values()))
        carried += 1
        print(f"pack_wheel: {name} carried once from "
              f"{'/'.join(next(iter(host_payloads)))}, sha256 {next(iter(digests.values()))[:12]}, "
              f"byte-identical across {len(host_payloads)} legs")
    print(f"pack_wheel: {carried} of {len(HOST_NAMES)} host bindings carried")
    # THE MERGED HOST INDEX, one copy beside the merged host file, held to the
    # same rule: every set that carries host bindings carries the same bytes.
    host_index_rel = f"host/{MERGED_INDEX}"
    host_indexes = {(s.vendor, s.arch): s.indexes[host_index_rel] for s in sets if host_index_rel in (s.indexes or {})}
    if host_indexes:
        digests = {k: sha(p).hex() for k, p in host_indexes.items()}
        carrying = [s for s in sets if s.hosts]
        if len(set(digests.values())) != 1 or len(host_indexes) != len(carrying):
            raise SystemExit(
                "pack_wheel: the architecture legs disagree on the merged host index (or not every leg "
                "that carries host bindings carries it): "
                + ", ".join(f"{v}/{a}={d[:12]}" for (v, a), d in sorted(digests.items())))
        entries[f"mojolearn/{host_index_rel}"] = next(iter(host_indexes.values()))

    # THE IDENTITY PAYLOAD (the packaging lane, 2026-09-14; docs/VERIFY.md),
    # the same files packaging/macos/build_release_wheel.sh copies into the
    # package tree: the one card comparator and the reference card directory
    # for `python -m mojolearn verify`, and tools/identity_break.py, the three
    # training GPU columns the manifest names and a commit witness for
    # `python -m mojolearn identity`. Read straight from their single sources
    # in this checkout; the wheel has no second implementation of anything.
    entries["mojolearn/_identity_trace_diff.py"] = REPO / "tools" / "identity_trace_diff.py"
    entries["mojolearn/_identity_break.py"] = REPO / "tools" / "identity_break.py"
    # The expansion lanes' identity lanes (lane/algos-prep, 2026-09-27): the
    # harness copy loads `_identity_lane_<lane>.py` beside itself.
    for fragment in sorted((REPO / "tools" / "identity_lanes").glob("*.py")):
        entries[f"mojolearn/_identity_lane_{fragment.stem}.py"] = fragment
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

    for source_set in sets:
        for rel, path in source_set.files.items():
            if rel.endswith("/" + gpu_plugins.BASELINE_MANIFEST):
                if json.loads(path.read_text())["source_commit"] != witness:
                    raise SystemExit("pack_wheel: the PTX set and core source commits differ")

    dist = f"mojolearn-{version}.dist-info"
    tag = f"py3-none-{plat}"
    generated = {
        f"{dist}/METADATA": metadata_text(proj, readme).encode(),
        f"{dist}/WHEEL": wheel_file_text(tag).encode(),
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

    # The split core's METADATA is the combined wheel's plus exactly the two
    # exact plugin requirements (core_project, written by write_split).

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
            print("  (a Linux classifier or a version bump are expected lines;"
                  " anything else is drift)")

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not split:
        whl = out / f"mojolearn-{version}-{tag}.whl"
        built = [(write_wheel(whl, entries, generated, dist), None)]
    else:
        built = write_split(out, kinds, entries, generated, dist, proj, version, tag, inventory, sets,
                            readme, bundled_ptx=bundle)

    # THE WHEEL SIZE CHECK, FIRST (Andrew 2026-10-10): before the audits (split_audit
    # refuses an oversize wheel too, with no per-set sizes), so the failure names them.
    oversize = oversize_report(built)
    if oversize:
        wheels = {}
        for whl, vendor in built:
            distribution = kind_distribution(vendor)
            limit = gpu_plugins.wheel_size_limit(distribution)
            wheels[whl.name] = dict(distribution=distribution, compressed_bytes=whl.stat().st_size,
                                    pypi_limit_bytes=limit, over_limit=whl.stat().st_size > limit,
                                    payloads=wheel_payload_sizes(whl))
        (out / f"SIZES-{version}-linux.json").write_text(json.dumps(
            dict(wheels=wheels, pypi_limit_bytes=PYPI_LIMIT, over_limit=True, profile=a.profile), indent=2))
        print("\n".join(["", "pack_wheel: WHEEL OVER PyPI's 100 MiB FILE LIMIT. STOP: this release cannot publish."]
                        + oversize), file=sys.stderr)
        return 1

    try:
        if _gates:
            # Finalize cached runtime closures too; never trust an older set to
            # be libm-free. A plugin carries no runtime directory, so it is
            # AUDITED (every binary it ships, the same audit the combined wheel
            # got) and not rewritten: its bytes stay exactly the sets' bytes.
            import subprocess
            for whl, vendor in built:
                command = [sys.executable, str(REPO / "packaging/portable_math/wheel.py"), str(whl)]
                if vendor is not None:
                    command.append("--audit-only")
                elif a.portable_math_helper:
                    command += ["--helper", str(a.portable_math_helper)]
                subprocess.run(command, check=True)

            # Audit independently of the package allow-list above, so adding an API
            # without updating packaging fails at build time rather than after upload.
            from wheel_api_audit import audit
            core_wheels = [w for w, vendor in built if vendor is None]
            if core_wheels:
                surface = audit(core_wheels)
                (out / f"API-{version}-linux.json").write_text(json.dumps(surface, indent=2) + "\n")
                if not surface['wheels'][0]['source_payload_complete']:
                    raise SystemExit('pack_wheel: incomplete source/API payload; see API report')
        if split:
            # EACH PLUGIN HOLDS EXACTLY ITS SETS AND THE CORE HOLDS NONE, the pins
            # are exact both ways (the core requires both plugins at its version,
            # each plugin the core), no extras, and no file is in two wheels
            # (wheel_api_audit).
            from wheel_api_audit import split_audit
            report = split_audit([w for w, _ in built])
            (out / f"SPLIT-{version}-linux.json").write_text(json.dumps(report, indent=2) + "\n")
            if report["problems"]:
                raise SystemExit("pack_wheel: the split wheels fail their audit:\n  "
                                 + "\n  ".join(report["problems"]))
    except BaseException:
        for whl, _ in built:
            whl.unlink(missing_ok=True)
        raise

    per_set = {}
    for s in sets:
        manifest = s.manifest
        per_set[f"{s.vendor}/{s.arch}"] = {
            "extensions_bytes": manifest["bytes_extensions"],
            "runtime_libs_bytes": manifest["bytes_staged_libs"],
            "driver_libs_not_staged": manifest["driver_libs_not_staged"],
        }
    if not split:
        whl = built[0][0]
        size = whl.stat().st_size
        sizes = {
            "wheel": str(whl), "compressed_bytes": size,
            "compressed_mb": round(size / 1e6, 2),
            "pypi_limit_bytes": PYPI_LIMIT, "over_limit": size > PYPI_LIMIT,
            "libs_layout": "shared mojolearn/.libs" if shared else "per-vendor <vendor>/.libs",
            "sets": per_set, "tag": tag,
        }
    else:
        wheels = {}
        for whl, vendor in built:
            size = whl.stat().st_size
            distribution = kind_distribution(vendor)
            limit = gpu_plugins.wheel_size_limit(distribution)
            wheels[whl.name] = {
                "distribution": distribution,
                "compressed_bytes": size, "compressed_mb": round(size / 1e6, 2),
                "pypi_limit_bytes": limit, "over_limit": size > limit,
                "payloads": wheel_payload_sizes(whl)}
        sizes = {
            "wheels": wheels, "pypi_limit_bytes": PYPI_LIMIT,
            "over_limit": any(w["over_limit"] for w in wheels.values()),
            "libs_layout": "shared mojolearn/.libs (core)", "sets": per_set, "tag": tag,
            "profile": a.profile,
        }
    (out / f"SIZES-{version}-linux.json").write_text(json.dumps(sizes, indent=2))
    print(json.dumps(sizes, indent=2))
    if sizes["over_limit"]:
        # THE WHEEL SIZE CHECK RUNS HERE, AT PACK TIME (Andrew 2026-10-10), and
        # fails loudly: PyPI refuses any file over 100 MiB, so an oversize wheel
        # must stop the release before any column or publish step runs.
        rows = sizes["wheels"].items() if split else [(pathlib.Path(sizes["wheel"]).name, dict(
            compressed_bytes=sizes["compressed_bytes"], pypi_limit_bytes=PYPI_LIMIT, over_limit=True,
            payloads=wheel_payload_sizes(pathlib.Path(sizes["wheel"]))))]
        lines = ["", "pack_wheel: WHEEL OVER PyPI's 100 MiB FILE LIMIT. STOP: this release cannot publish.",
                 f"  report: {out / f'SIZES-{version}-linux.json'}"]
        for name, w in rows:
            lines.append(f"  {'OVER' if w['over_limit'] else 'ok  '} {name}: {w['compressed_bytes'] / 2**20:.1f} MiB "
                         f"(limit {w['pypi_limit_bytes'] / 2**20:.0f} MiB)")
            for payload, nbytes in sorted(w["payloads"].items(), key=lambda kv: -kv[1]):
                lines.append(f"         {payload}: {nbytes / 2**20:.1f} MiB compressed")
        print("\n".join(lines), file=sys.stderr)
        return 1
    return 0


def oversize_report(built):
    """Report lines for every built wheel when any is over its PyPI limit, else []."""
    rows = []
    for whl, vendor in built:
        distribution = kind_distribution(vendor)
        rows.append((whl, whl.stat().st_size, gpu_plugins.wheel_size_limit(distribution)))
    if not any(size > limit for _, size, limit in rows):
        return []
    lines = []
    for whl, size, limit in rows:
        lines.append(f"  {'OVER' if size > limit else 'ok  '} {whl.name}: {size / 2**20:.1f} MiB (limit {limit / 2**20:.0f} MiB)")
        for payload, nbytes in sorted(wheel_payload_sizes(whl).items(), key=lambda kv: -kv[1]):
            lines.append(f"         {payload}: {nbytes / 2**20:.1f} MiB compressed")
    return lines


def wheel_payload_sizes(whl):
    """{payload: compressed bytes} of one wheel: each architecture set
    (mojolearn/<dir>/<arch>, e.g. cuda_native/sm_89, cuda_ptx/sm_80), the
    shared runtime and everything else, from the zip's own member sizes."""
    out = {}
    with zipfile.ZipFile(whl) as z:
        for info in z.infolist():
            parts = info.filename.split("/")
            if (len(parts) > 3 and parts[0] == "mojolearn"
                    and parts[1] in ("cuda", "hip", "cuda_native", "hip_native", gpu_plugins.PTX_DIRECTORY)):
                key = "/".join(parts[1:3])
            elif len(parts) > 2 and parts[0] == "mojolearn" and parts[1] == ".libs":
                key = ".libs (runtime)"
            else:
                key = "other"
            out[key] = out.get(key, 0) + info.compress_size
    return out


def split_kinds(text, vendors_given, keys=None):
    """The wheels a split profile emits, in WHEEL_KINDS order: `--wheels`
    when given, else the core plus the plugin of every vendor a set was given
    for, plus the meta `mojolearn` wheel when every vendor was given. A
    plugin (or the meta wheel) asked for with no set of its vendor(s) is refused."""
    if not text:
        kinds = [gpu_plugins.CORE_PROFILE] + [row["profile"] for row in gpu_plugins.distribution_rows()
                                             if row["vendor"] in vendors_given]
        if set(gpu_plugins.vendors()) <= set(vendors_given):
            kinds.append(gpu_plugins.META_PROFILE)
    else:
        kinds = [k.strip() for k in text.split(",") if k.strip()]
        unknown = [k for k in kinds if k not in WHEEL_KINDS]
        if unknown or not kinds or len(set(kinds)) != len(kinds):
            raise SystemExit(f"pack_wheel: --wheels {text!r}: name each of {', '.join(WHEEL_KINDS)} "
                             "at most once, nothing else")
    for kind in kinds:
        if kind == gpu_plugins.META_PROFILE:
            missing = [v for v in gpu_plugins.vendors() if v not in vendors_given]
            if missing:
                raise SystemExit(f"pack_wheel: --wheels names {kind}, which requires every vendor, "
                                 f"but no {', '.join(missing)} set was given")
            continue
        if kind != gpu_plugins.CORE_PROFILE and gpu_plugins.by_profile(kind) not in vendors_given:
            raise SystemExit(f"pack_wheel: --wheels names {kind} but no "
                             f"{gpu_plugins.by_profile(kind)} set was given")
    if keys is not None:
        for kind in kinds:
            if kind in gpu_plugins.PAYLOADS:
                row = gpu_plugins.PAYLOADS[kind]
                arches = [a for v, a in keys if v == row["vendor"] and a in row["arches"]]
                if not gpu_plugins.valid_arches(kind, arches):
                    raise SystemExit(f"pack_wheel: {kind} requires one set per registered architecture slot; found {arches}")
                if (row.get("ptx") and row.get("ptx_shipped", True)
                        and (row["vendor"], row["ptx"]["arch"]) not in keys):
                    raise SystemExit(f"pack_wheel: {kind} requires one set per registered architecture slot, "
                                     f"the PTX slot {row['vendor']}/{row['ptx']['arch']} included; found {arches}")
    return tuple(k for k in WHEEL_KINDS if k in kinds)


def split_payload(entries, generated, dist):
    """Partition by payload profile, remapping legacy roots at equal depth.

    None owns core files; each vendor profile owns its architecture sets.
    """
    parts = {}
    for arc, src in entries.items():
        parts.setdefault(gpu_plugins.member_payload(arc), ({}, {}))[0][gpu_plugins.installed_member(arc)] = src
    for arc, data in generated.items():
        if arc.startswith(dist + "/"):
            continue
        parts.setdefault(gpu_plugins.member_payload(arc), ({}, {}))[1][gpu_plugins.installed_member(arc)] = data
    return parts


def write_split(out, kinds, entries, generated, dist, proj, version, tag, inventory, sets, readme, bundled_ptx=None):
    """Write the split wheels `kinds` asks for from the combined payload.
    Returns [(wheel path, profile or None for the core), ...]. Native bytes
    are preserved while GPU roots move to disjoint equal-depth directories.
    The core's .dist-info is mojolearn_core-<v>.dist-info (2026-10-20); the
    meta `mojolearn` wheel carries only its .dist-info."""
    parts = split_payload(entries, generated, dist)
    core_entries, core_payload = parts.get(None, ({}, {}))
    licenses = {k: v for k, v in generated.items() if k.startswith(f"{dist}/licenses/")}
    built = []

    def payload_doc(role, distribution, members):
        doc = dict(inventory)
        doc["split"] = dict(role=role, distribution=distribution, profile=RELEASE_SPLIT_PROFILE,
                            native_members=sorted(n for n in members
                                                  if n.endswith(".so") or "/.libs/" in n))
        return (json.dumps(doc, sort_keys=True, indent=2) + "\n").encode()

    try:
        for kind in kinds:
            if kind == gpu_plugins.CORE_PROFILE:
                cdist = f"{gpu_plugins.CORE_WHEEL_NAME}-{version}.dist-info"
                core_generated = {}
                for arc, data in generated.items():
                    if not arc.startswith(dist + "/"):
                        continue
                    if arc == f"{dist}/LINUX_PAYLOAD.json":
                        data = payload_doc(kind, gpu_plugins.CORE_DISTRIBUTION, core_entries)
                    elif arc == f"{dist}/METADATA":
                        data = metadata_text(core_project(proj, version), CORE_README_NOTE + readme).encode()
                    core_generated[cdist + arc[len(dist):]] = data
                core_generated[f"{cdist}/{gpu_plugins.CORE_MARKER}"] = (
                    json.dumps(gpu_plugins.core_marker(version), sort_keys=True, indent=2) + "\n").encode()
                core_generated.update(core_payload)
                whl = out / f"{gpu_plugins.CORE_WHEEL_NAME}-{version}-{tag}.whl"
                built.append((write_wheel(whl, core_entries, core_generated, cdist), None))
                continue
            if kind == gpu_plugins.META_PROFILE:
                # THE LINUX `mojolearn` WHEEL (2026-10-20): metadata only. Its
                # requirements install the core and both vendors; its marker is
                # an aggregate, which audit.sh admits without auditwheel.
                mdist = f"{gpu_plugins.META_WHEEL_NAME}-{version}.dist-info"
                mproj, mreadme = meta_project(proj, version)
                mgen = {
                    f"{mdist}/METADATA": metadata_text(mproj, mreadme).encode(),
                    f"{mdist}/WHEEL": wheel_file_text(tag).encode(),
                }
                if inventory is not None:
                    mgen[f"{mdist}/LINUX_PAYLOAD.json"] = payload_doc(kind, gpu_plugins.META_DISTRIBUTION, {})
                for arc, data in licenses.items():
                    mgen[mdist + arc[len(dist):]] = data
                mgen[f"{mdist}/{gpu_plugins.PLUGIN_MARKER}"] = (json.dumps(
                    gpu_plugins.meta_marker(version), sort_keys=True, indent=2) + "\n").encode()
                whl = out / f"{gpu_plugins.META_WHEEL_NAME}-{version}-{tag}.whl"
                built.append((write_wheel(whl, {}, mgen, mdist), kind))
                continue
            vendor = gpu_plugins.by_profile(kind)
            row = gpu_plugins.package(kind)
            ventries, vpayload = parts.get(kind, ({}, {}))
            if not ventries:
                raise SystemExit(f"pack_wheel: {row['distribution']} would be empty; no {vendor} set")
            arches = sorted({s.arch for s in sets if s.vendor == vendor and
                             (s.arch in row["arches"])})
            pproj, preadme = plugin_project(proj, kind, version, arches)
            pdist = f"{row['wheel_name']}-{version}.dist-info"
            pgen = {
                f"{pdist}/METADATA": metadata_text(pproj, preadme).encode(),
                f"{pdist}/WHEEL": wheel_file_text(tag).encode(),
            }
            if inventory is not None:
                pgen[f"{pdist}/LINUX_PAYLOAD.json"] = payload_doc(kind, row["distribution"], ventries)
            for arc, data in licenses.items():
                pgen[pdist + arc[len(dist):]] = data
            try:
                marker = gpu_plugins.package_marker(kind, version, arches,
                                                    bundled_ptx=bundled_ptx if row.get("ptx") or row["code_format"] == "ptx" else None)
            except ValueError as exc:
                raise SystemExit(f"pack_wheel: {row['distribution']}: {exc}") from exc
            pgen[f"{pdist}/{gpu_plugins.PLUGIN_MARKER}"] = (json.dumps(
                marker, sort_keys=True, indent=2) + "\n").encode()
            pgen.update(vpayload)
            whl = out / f"{row['wheel_name']}-{version}-{tag}.whl"
            built.append((write_wheel(whl, ventries, pgen, pdist), kind))
    except BaseException:
        for whl, _ in built:
            whl.unlink(missing_ok=True)
        raise
    for whl, vendor in built:
        print(f"pack_wheel: wrote {whl.name} ({'core' if vendor is None else 'meta' if vendor == gpu_plugins.META_PROFILE else vendor + ' plugin'})")
    return built

if __name__ == "__main__":
    sys.exit(main())
