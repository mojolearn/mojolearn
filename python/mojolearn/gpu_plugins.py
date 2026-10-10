# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Linux vendor distributions and architecture file ownership.

The core pins mojolearn-nvidia and mojolearn-amd automatically. Each vendor
wheel contains its registered architecture sets; architecture selection
happens in the loader. mojolearn-nvidia carries the PTX set
(`cuda_ptx/sm_80`) as a regular slot beside its native sets (Andrew
2026-10-10: PTX is a normal target; no flag). Native directory depth stays
unchanged to preserve binding RUNPATHs.
"""

#: The NVIDIA PTX slot: one rounding-pinned PTX build for compute capability
#: 8.0, compiled by the driver on any NVIDIA GPU of capability 8.0 or newer
#: that has no native set in the wheel. Its directory sits at the native
#: depth (mojolearn/cuda_ptx/sm_80/...), so every RUNPATH resolves the same.
PTX_ARCH = "sm_80"
PTX_DIRECTORY = "cuda_ptx"

PLUGINS = {
    "cuda": dict(distribution="mojolearn-nvidia", wheel_name="mojolearn_nvidia",
        profile="nvidia", label="NVIDIA (CUDA)", vendor="cuda", role="vendor",
        # 0.8.37 (Andrew 2026-10-10): the release ships sm_89 only. With sm_90a the wheel is 134.6 MiB, over
        # PyPI's 100 MiB file limit, so the Hopper slot ("sm_90", "sm_90a") is not required until PyPI raises the
        # mojolearn-nvidia limit; sm_90/sm_90a stay registered architectures, so a wheel that carries them loads.
        # Blackwell is registered the same way (Andrew 2026-10-10): sm_100/sm_100a (B200/GB200),
        # sm_103/sm_103a (B300), sm_120/sm_120a (RTX PRO 6000 / RTX 50) and sm_121/sm_121a (DGX
        # Spark). A wheel carrying one of those native sets loads it; without one, Blackwell runs
        # the PTX slot. Andrew 2026-10-10: PTX is a normal target; no flag.
        arches=("sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_103", "sm_103a",
                "sm_120", "sm_120a", "sm_121", "sm_121a"), slots=(("sm_89",),),
        # Andrew 2026-10-10: PTX is a normal target; no flag. The PTX slot is required like sm_89.
        ptx=dict(arch=PTX_ARCH, directory=PTX_DIRECTORY),
        directory="cuda_native", code_format="native", release_enabled=True),
    "hip": dict(distribution="mojolearn-amd", wheel_name="mojolearn_amd",
        profile="amd", label="AMD (ROCm/HIP)", vendor="hip", role="vendor",
        arches=("gfx942",), slots=(("gfx942",),), directory="hip_native", ptx=None,
        code_format="native", release_enabled=True),
}
PAYLOADS = {row["profile"]: row for row in PLUGINS.values()}  # glue: index vendor package metadata by release profile


def distribution_rows(include_experimental=False):
    """Every vendor distribution. There is no experimental payload any more
    (the PTX set is a slot of mojolearn-nvidia); the argument is kept for
    callers that still pass it."""
    return tuple(PAYLOADS.values())


def valid_arches(profile, arches):
    row = PAYLOADS[profile]
    return (len(arches) == len(set(arches)) and set(arches) <= set(row["arches"])
            and all(len(set(slot) & set(arches)) == 1 for slot in row["slots"]))  # glue: validate one installed set per registered hardware slot


def wheel_size_limit(distribution):
    """PyPI's per-file upload limit, the same for every distribution: 100 MiB.
    No larger allowance has been granted for any mojolearn project (0.8.37:
    the sm_89 + sm_90a NVIDIA wheel was 134.6 MiB and could not publish)."""
    return 100 * 1024**2


def package(profile):
    return next(row for row in distribution_rows(include_experimental=True) if row["profile"] == profile)  # glue: look up one package profile in the distribution registry


def payload_for(vendor, arch):
    for row in PAYLOADS.values():  # glue: look up metadata for a vendor and architecture target
        if row["vendor"] == vendor and (arch in row["arches"] or is_ptx_arch(row, arch)):
            return row
    raise KeyError((vendor, arch))


def is_ptx_arch(row, arch):
    """Is `arch` the PTX slot of the vendor row `row`?"""
    return bool(row.get("ptx")) and arch == row["ptx"]["arch"]


def set_directory(row, arch):
    """The installed directory of one architecture set of a vendor row:
    the PTX directory for the PTX slot, the native directory otherwise."""
    return row["ptx"]["directory"] if is_ptx_arch(row, arch) else row["directory"]


def required_sets(vendor):
    """{(vendor, arch)} a complete vendor wheel carries: the first spelling
    of every native slot, plus the PTX slot when the vendor has one."""
    row = PLUGINS[vendor]
    out = {(vendor, slot[0]) for slot in row["slots"]}  # glue: one entry per registered slot
    if row.get("ptx"):
        out.add((vendor, row["ptx"]["arch"]))
    return out


def native_directory(vendor):
    return {"cuda": "cuda_native", "hip": "hip_native"}[vendor]


def payload_requirements(vendor, version):
    """Vendor projects have no architecture-package dependencies."""
    return []


def member_payload(arcname):
    parts = arcname.split("/")
    if len(parts) < 4 or parts[0] != "mojolearn":
        return None
    for key, row in PAYLOADS.items():  # glue: map an archive member path to its registered package owner
        if parts[1] in (row["vendor"], row["directory"]) and parts[2] in row["arches"]:
            return key
        if (row.get("ptx") and parts[1] in (row["vendor"], row["ptx"]["directory"])
                and parts[2] == row["ptx"]["arch"]):
            return key
    if parts[1] in (*PLUGINS, "cuda_native", "hip_native", "cuda_ptx"):
        raise ValueError(f"unregistered GPU payload member: {arcname}")
    return None


def installed_member(arcname):
    owner = member_payload(arcname)
    if owner is None:
        return arcname
    parts = arcname.split("/")
    parts[1] = set_directory(PAYLOADS[owner], parts[2])
    return "/".join(parts)


#: The core distribution, and the name of the profile that packs it.
CORE_DISTRIBUTION = "mojolearn"
CORE_PROFILE = "core-linux"

#: A file in the core's .dist-info that says "this install is the split
#: core; its GPU sets come from plugins". Its absence means the old combined
#: wheel, a macOS wheel or a source checkout, where nothing here applies.
CORE_MARKER = "gpu_plugins.json"
#: The same, in each plugin's .dist-info: which vendor and architectures it
#: carries and the core version it was packed with.
PLUGIN_MARKER = "gpu_plugin.json"
CORE_SCHEMA = "mojolearn.gpu-plugins.v3"
PLUGIN_SCHEMA = "mojolearn.gpu-plugin.v3"


def vendors():
    """The vendor directories a plugin can ship, in table order."""
    return tuple(PLUGINS)


def plugin(vendor):
    """The table row of `vendor` ('cuda' or 'hip')."""
    return PLUGINS[vendor]


def by_profile(profile):
    """vendor for a plugin profile name ('nvidia' -> 'cuda', 'amd' -> 'hip')."""
    for vendor, row in PLUGINS.items():  # glue: one row per GPU vendor
        if row["profile"] == profile:
            return vendor
    if profile in PAYLOADS:
        return PAYLOADS[profile]["vendor"]
    raise KeyError(profile)


def member_vendor(arcname):
    """Which plugin owns a wheel member: the vendor for
    `mojolearn/<vendor>/...`, None for everything the core ships."""
    parts = arcname.split("/")
    if len(parts) > 2 and parts[0] == "mojolearn":
        if parts[1] in PLUGINS:
            return parts[1]
        for vendor in PLUGINS:  # glue: classify a package directory by vendor metadata
            row = PLUGINS[vendor]
            if parts[1] == native_directory(vendor) or (row.get("ptx") and parts[1] == row["ptx"]["directory"]):
                return vendor
    return None


def core_requirements(version):
    """The Linux core's Requires-Dist values on its GPU plugins, in table
    order: every plugin, at the core's own version exactly. No environment
    marker: the split core is a manylinux x86_64 wheel only, and a
    `sys_platform` marker is evaluated against the RESOLVING interpreter, so
    `pip download --platform manylinux...` from a Mac would skip both."""
    return [f"{row['distribution']}=={version}" for row in PLUGINS.values()]  # glue: one row per GPU vendor


def reinstall_command(version=None):
    """The pip command that repairs an incomplete split install: the core at
    this version, which brings both plugins through its exact requirements."""
    pin = f"=={version}" if version else ""
    return f'pip install --force-reinstall "{CORE_DISTRIBUTION}{pin}"'


def core_marker(version):
    """The core's marker document."""
    return {"schema": CORE_SCHEMA, "version": version,
            "plugins": {v: {"distribution": r["distribution"]} for v, r in PLUGINS.items()}}  # glue: one row per GPU vendor


def plugin_marker(vendor, version, arches, bundled_ptx=None):
    """`arches` are the NATIVE architecture directories. A vendor with a PTX
    slot (NVIDIA) must name its PTX set's manifest digest in `bundled_ptx`
    ({"manifest_sha256": hex}); the PTX slot is part of every such wheel."""
    row = PLUGINS[vendor]
    marker = {"schema": PLUGIN_SCHEMA, "role": "vendor", "vendor": vendor, "version": version,
            "distribution": row["distribution"], "requires": f"{CORE_DISTRIBUTION}=={version}",
            "arches": sorted(arches), "directory": row["directory"], "code_format": "native"}  # glue: canonicalize architecture identifiers in metadata
    if row.get("ptx"):
        if bundled_ptx is None:
            raise ValueError(f"{row['distribution']} carries the PTX slot; its PTX manifest digest is required")
        validate_bundle_descriptor(bundled_ptx)
        marker["bundled_ptx"] = dict(bundled_ptx)
        marker["ptx"] = dict(row["ptx"])
    elif bundled_ptx is not None:
        raise ValueError("only the NVIDIA vendor package carries a PTX slot")
    return marker


def package_requirements(profile, version):
    if profile == CORE_PROFILE:
        return core_requirements(version)
    package(profile)  # Validate that the profile is registered.
    return [f"{CORE_DISTRIBUTION}=={version}"]


def package_marker(profile, version, arches=(), bundled_ptx=None):
    if profile == CORE_PROFILE:
        if bundled_ptx is not None:
            raise ValueError("only the NVIDIA vendor package carries a PTX slot")
        return core_marker(version)
    package(profile)  # Validate that the profile is registered.
    return plugin_marker(by_profile(profile), version, arches, bundled_ptx=bundled_ptx)


BASELINE_MANIFEST = "PTX_BASELINE.json"
#: The PTX set's build manifest schema (packaging/linux/ptx_baseline.py).
#: v2 (Andrew 2026-10-10: PTX is a normal target; no flag) drops the
#: experimental / identical_qualified / qualification_required fields: the
#: PTX set's identity is decided by its column in the reference table, the
#: same way as every native set's.
PTX_MANIFEST_SCHEMA = "mojolearn.ptx-set.v2"
PTX_CODE_FORMAT = "ptx"


def validate_baseline_manifest(doc, files):
    """Validate the PTX set's build manifest against the actual relative-file
    SHA256 values of its GPU bindings. Artifact ownership and build format
    only; identity is the reference table's PTX column."""
    import re
    if (not isinstance(doc, dict) or doc.get("schema") != PTX_MANIFEST_SCHEMA
            or doc.get("code_format") != PTX_CODE_FORMAT or doc.get("vendor") != "cuda"
            or doc.get("target") != PTX_ARCH or doc.get("min_compute_capability") != [8, 0]
            or doc.get("errors") != []
            or not re.fullmatch(r"[0-9a-f]{40}", doc.get("source_commit", ""))):
        raise ValueError("invalid PTX set manifest")
    rows = doc.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):  # glue: validate JSON file-manifest record shapes
        raise ValueError("PTX manifest files must be a list of file records")
    declared = {}
    for row in rows:  # glue: validate file-manifest names and recorded SHA256 metadata
        name = row.get("file", "")
        if (not isinstance(name, str) or not name or name.startswith("/")
                or ".." in name.split("/") or name in declared):
            raise ValueError("invalid or duplicate PTX manifest file")
        mode = name.split("/", 1)[0] if "/" in name else "fast"
        if mode not in ("fast", "deterministic", "identical") or row.get("numeric_mode") != mode:
            raise ValueError(f"missing or mismatched numerical-mode evidence for {name}")
        modules = row.get("ptx_modules")
        if not isinstance(modules, list) or not all(isinstance(module, dict) for module in modules):  # glue: validate JSON PTX evidence record shapes
            raise ValueError(f"PTX module evidence must be a list of records for {name}")
        if not modules:
            prefix = name.rsplit("/", 1)[0] + "/" if "/" in name else ""
            expected = {prefix + "_mojolearn_rf.so", prefix + "_mojolearn_gbdt.so"}
            delegates = row.get("delegates", [])
            if (name != prefix + "_mojolearn_x_trees.so"
                    or not isinstance(delegates, list)
                    or not all(isinstance(delegate, dict) for delegate in delegates)  # glue: validate JSON delegated-file record shapes
                    or {d.get("file") for d in delegates} != expected  # glue: compare declared delegated filenames against allowed file ownership
                    or len(delegates) != len(expected)
                    or any(d.get("sha256") != files.get(d.get("file")) for d in delegates)  # glue: match delegated-file digest strings to the file manifest
                    or any(not next((r.get("ptx_modules") for r in doc["files"]  # glue: look up delegated-file PTX evidence in manifest records
                                     if r.get("file") == target), None) for target in expected)):  # glue: check evidence exists for each declared delegate filename
                raise ValueError(f"missing PTX or registered delegation evidence for {name}")
        declared[name] = row.get("sha256")
    if not declared or declared != files:
        raise ValueError("PTX manifest does not match the complete GPU binding bytes")
    return doc


BUNDLED_PTX_ROOT = f"mojolearn/{PTX_DIRECTORY}/{PTX_ARCH}"


def validate_bundle_descriptor(descriptor):
    """The vendor marker binds the PTX set's manifest by its SHA256."""
    import re
    if (not isinstance(descriptor, dict) or set(descriptor) != {"manifest_sha256"}
            or not isinstance(descriptor["manifest_sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", descriptor["manifest_sha256"])):
        raise ValueError("invalid PTX slot digest descriptor")


def owns_member(profile, member, bundled_ptx=None):
    """Does the package `profile` own the wheel member `member`? The PTX
    slot is owned by the NVIDIA vendor package like its native sets."""
    return member_payload(member) == profile


def validate_bundled_ptx(descriptor, manifest_bytes, files):
    """Validate the PTX slot's transported bytes against the vendor marker's
    descriptor; return the parsed manifest.

    files is the complete relative GPU .so SHA256 map under cuda_ptx/sm_80.
    """
    import hashlib
    import json
    validate_bundle_descriptor(descriptor)
    if hashlib.sha256(manifest_bytes).hexdigest() != descriptor["manifest_sha256"]:
        raise ValueError("PTX manifest bytes differ from the vendor marker")
    manifest = json.loads(manifest_bytes)
    validate_baseline_manifest(manifest, files)
    if manifest.get("source_dirty") is not False:
        raise ValueError("the PTX set needs a clean source manifest")
    return manifest
