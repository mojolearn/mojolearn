# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Linux GPU distributions and file ownership, shared by packer and loader.

The core requires both vendor aggregates; each aggregate requires its native
architecture payloads. Payloads require the exact core version. Thus ordinary
``pip install mojolearn`` still installs every released GPU target. A payload
owns only its architecture, and aggregates own no kernel files.

Native payloads use cuda_native/ and hip_native/ rather than the old vendor
wheel roots. Upgrading an old vendor wheel cannot uninstall the new payload's
files. Directory depth is unchanged, preserving each binding's relative RUNPATH.
Experimental PTX is deliberately absent from the released payload registry.
"""

#: vendor directory -> the plugin that ships it. `vendor` is the directory
#: name under mojolearn/ and the string `<prefix>_vendor()` answers.
PLUGINS = {
    "cuda": {
        "distribution": "mojolearn-nvidia",
        "wheel_name": "mojolearn_nvidia",
        "profile": "nvidia",
        "label": "NVIDIA (CUDA)",
        "role": "aggregate",
    },
    "hip": {
        "distribution": "mojolearn-amd",
        "wheel_name": "mojolearn_amd",
        "profile": "amd",
        "label": "AMD (ROCm/HIP)",
        "role": "aggregate",
    },
}

# Payload project names are centralized; external project registration is a
# separate release prerequisite, not implied by this registry.
PAYLOADS = {
    "nvidia-sm89": dict(distribution="mojolearn-nvidia-sm89", wheel_name="mojolearn_nvidia_sm89",
        profile="nvidia-sm89", label="NVIDIA Ada native", vendor="cuda", arches=("sm_89",),
        directory="cuda_native", code_format="native", role="payload"),
    "nvidia-sm90": dict(distribution="mojolearn-nvidia-sm90", wheel_name="mojolearn_nvidia_sm90",
        profile="nvidia-sm90", label="NVIDIA Hopper native", vendor="cuda", arches=("sm_90", "sm_90a"),
        directory="cuda_native", code_format="native", role="payload"),
    "amd-gfx942": dict(distribution="mojolearn-amd-gfx942", wheel_name="mojolearn_amd_gfx942",
        profile="amd-gfx942", label="AMD gfx942 native", vendor="hip", arches=("gfx942",),
        directory="hip_native", code_format="native", role="payload"),
}


for _row in (*PLUGINS.values(), *PAYLOADS.values()):
    _row["release_enabled"] = True
PAYLOADS["nvidia-ptx80"] = dict(distribution="mojolearn-nvidia-ptx80", wheel_name="mojolearn_nvidia_ptx80",
    profile="nvidia-ptx80", label="NVIDIA experimental PTX baseline", vendor="cuda", arches=("sm_80",),
    directory="cuda_ptx", code_format="ptx-baseline", role="payload", release_enabled=False)


def distribution_rows(include_experimental=False):
    return tuple(row for row in (*PLUGINS.values(), *PAYLOADS.values())
                 if include_experimental or row["release_enabled"])


def package(profile):
    return next(row for row in distribution_rows(include_experimental=True) if row["profile"] == profile)


def payload_for(vendor, arch):
    for row in PAYLOADS.values():
        if row["vendor"] == vendor and arch in row["arches"]:
            return row
    raise KeyError((vendor, arch))


def native_directory(vendor):
    return {"cuda": "cuda_native", "hip": "hip_native"}[vendor]


def payload_requirements(vendor, version):
    return [f"{row['distribution']}=={version}" for row in PAYLOADS.values()
            if row["vendor"] == vendor and row["release_enabled"]]


def member_payload(arcname):
    parts = arcname.split("/")
    if len(parts) < 4 or parts[0] != "mojolearn":
        return None
    for key, row in PAYLOADS.items():
        if parts[1] in (row["vendor"], row["directory"]) and parts[2] in row["arches"]:
            return key
    if parts[1] in (*PLUGINS, "cuda_native", "hip_native", "cuda_ptx"):
        raise ValueError(f"unregistered GPU payload member: {arcname}")
    return None


def installed_member(arcname):
    owner = member_payload(arcname)
    if owner is None:
        return arcname
    parts = arcname.split("/")
    parts[1] = PAYLOADS[owner]["directory"]
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
CORE_SCHEMA = "mojolearn.gpu-plugins.v2"
PLUGIN_SCHEMA = "mojolearn.gpu-plugin.v2"
PAYLOAD_MARKER = "gpu_payload.json"
PAYLOAD_SCHEMA = "mojolearn.gpu-payload.v1"


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
        for vendor in PLUGINS:
            if parts[1] == native_directory(vendor) or (vendor == "cuda" and parts[1] == "cuda_ptx"):
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


def plugin_marker(vendor, version, arches):
    return {"schema": PLUGIN_SCHEMA, "role": "aggregate", "vendor": vendor, "version": version,
            "distribution": PLUGINS[vendor]["distribution"],
            "requires": payload_requirements(vendor, version), "arches": [],
            "payloads": [r["distribution"] for r in PAYLOADS.values() if r["vendor"] == vendor and r["release_enabled"]]}


def payload_marker(profile, version, arches):
    row = PAYLOADS[profile]
    return {"schema": PAYLOAD_SCHEMA, "role": "payload", "vendor": row["vendor"],
            "version": version, "distribution": row["distribution"],
            "requires": f"{CORE_DISTRIBUTION}=={version}", "arches": sorted(arches),
            "directory": row["directory"], "code_format": row["code_format"]}


def package_requirements(profile, version):
    if profile == CORE_PROFILE:
        return core_requirements(version)
    row = package(profile)
    if row["role"] == "aggregate":
        return payload_requirements(by_profile(profile), version)
    return [f"{CORE_DISTRIBUTION}=={version}"]


def package_marker(profile, version, arches=()):
    if profile == CORE_PROFILE:
        return core_marker(version)
    if package(profile)["role"] == "aggregate":
        return plugin_marker(by_profile(profile), version, arches)
    return payload_marker(profile, version, arches)


BASELINE_MANIFEST = "PTX_BASELINE.json"


def validate_baseline_manifest(doc, files):
    """Validate a build manifest against actual relative-file SHA256 values.

    This establishes artifact ownership, not bitwise qualification. The latter
    stays false until a separate hardware/compiler qualification process.
    """
    import re
    if (not isinstance(doc, dict) or doc.get("schema") != "mojolearn.ptx-baseline.v1"
            or doc.get("code_format") != "ptx-baseline" or doc.get("vendor") != "cuda"
            or doc.get("target") != "sm_80" or doc.get("min_compute_capability") != [8, 0]
            or doc.get("experimental") is not True or doc.get("identical_qualified") is not False
            or doc.get("qualification_required") is not True or doc.get("errors") != []
            or not re.fullmatch(r"[0-9a-f]{40}", doc.get("source_commit", ""))):
        raise ValueError("invalid or qualified-as-production experimental PTX manifest")
    rows = doc.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("PTX manifest files must be a list of file records")
    declared = {}
    for row in rows:
        name = row.get("file", "")
        if (not isinstance(name, str) or not name or name.startswith("/")
                or ".." in name.split("/") or name in declared):
            raise ValueError("invalid or duplicate PTX manifest file")
        mode = name.split("/", 1)[0] if "/" in name else "fast"
        if mode not in ("fast", "deterministic", "identical") or row.get("numeric_mode") != mode:
            raise ValueError(f"missing or mismatched numerical-mode evidence for {name}")
        modules = row.get("ptx_modules")
        if not isinstance(modules, list) or not all(isinstance(module, dict) for module in modules):
            raise ValueError(f"PTX module evidence must be a list of records for {name}")
        if not modules:
            prefix = name.rsplit("/", 1)[0] + "/" if "/" in name else ""
            expected = {prefix + "_mojolearn_rf.so", prefix + "_mojolearn_gbdt.so"}
            delegates = row.get("delegates", [])
            if (name != prefix + "_mojolearn_x_trees.so"
                    or not isinstance(delegates, list)
                    or not all(isinstance(delegate, dict) for delegate in delegates)
                    or {d.get("file") for d in delegates} != expected
                    or len(delegates) != len(expected)
                    or any(d.get("sha256") != files.get(d.get("file")) for d in delegates)
                    or any(not next((r.get("ptx_modules") for r in doc["files"]
                                     if r.get("file") == target), None) for target in expected)):
                raise ValueError(f"missing PTX or registered delegation evidence for {name}")
        declared[name] = row.get("sha256")
    if not declared or declared != files:
        raise ValueError("PTX manifest does not match the complete GPU binding bytes")
    return doc
