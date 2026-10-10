# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Linux vendor distributions and native architecture file ownership.

The core pins mojolearn-nvidia and mojolearn-amd automatically. Each vendor
wheel contains its registered native architecture sets; architecture selection
still happens in the loader. Qualified PTX may be bundled in the NVIDIA wheel;
experimental PTX also remains a separate opt-in artifact.
Native directory depth stays unchanged to preserve binding RUNPATHs.
"""

PLUGINS = {
    "cuda": dict(distribution="mojolearn-nvidia", wheel_name="mojolearn_nvidia",
        profile="nvidia", label="NVIDIA (CUDA)", vendor="cuda", role="vendor",
        # 0.8.37 (Andrew 2026-10-10): the release ships sm_89 only. With sm_90a the wheel is 134.6 MiB, over
        # PyPI's 100 MiB file limit, so the Hopper slot ("sm_90", "sm_90a") is not required until PyPI raises the
        # mojolearn-nvidia limit; sm_90/sm_90a stay registered architectures, so a wheel that carries them loads.
        arches=("sm_89", "sm_90", "sm_90a"), slots=(("sm_89",),),
        directory="cuda_native", code_format="native", release_enabled=True),
    "hip": dict(distribution="mojolearn-amd", wheel_name="mojolearn_amd",
        profile="amd", label="AMD (ROCm/HIP)", vendor="hip", role="vendor",
        arches=("gfx942",), slots=(("gfx942",),), directory="hip_native",
        code_format="native", release_enabled=True),
}
PAYLOADS = {row["profile"]: row for row in PLUGINS.values()}  # glue: index vendor package metadata by release profile
PAYLOADS["nvidia-ptx80"] = dict(distribution="mojolearn-nvidia-ptx80", wheel_name="mojolearn_nvidia_ptx80",
    profile="nvidia-ptx80", label="NVIDIA experimental PTX baseline", vendor="cuda", arches=("sm_80",),
    slots=(("sm_80",),), directory="cuda_ptx", code_format="ptx-baseline", role="payload", release_enabled=False)


def distribution_rows(include_experimental=False):
    return tuple(row for row in PAYLOADS.values()  # glue: filter package registry records for release metadata
                 if include_experimental or row["release_enabled"])


def valid_arches(profile, arches):
    row = PAYLOADS[profile]
    return (len(arches) == len(set(arches)) and set(arches) <= set(row["arches"])
            and all(len(set(slot) & set(arches)) == 1 for slot in row["slots"]))  # glue: validate one installed set per registered hardware slot


def wheel_size_limit(distribution):
    # Requested project allowance, not evidence that PyPI has granted it.
    # Publication must confirm mojolearn-nvidia's 250 MiB allowance first.
    return (250 if distribution == "mojolearn-nvidia" else 100) * 1024**2


def package(profile):
    return next(row for row in distribution_rows(include_experimental=True) if row["profile"] == profile)  # glue: look up one package profile in the distribution registry


def payload_for(vendor, arch):
    for row in PAYLOADS.values():  # glue: look up metadata for a vendor and architecture target
        if row["vendor"] == vendor and arch in row["arches"]:
            return row
    raise KeyError((vendor, arch))


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
CORE_SCHEMA = "mojolearn.gpu-plugins.v3"
PLUGIN_SCHEMA = "mojolearn.gpu-plugin.v3"
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
        for vendor in PLUGINS:  # glue: classify a package directory by vendor metadata
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


def plugin_marker(vendor, version, arches, bundled_ptx=None):
    row = PLUGINS[vendor]
    marker = {"schema": PLUGIN_SCHEMA, "role": "vendor", "vendor": vendor, "version": version,
            "distribution": row["distribution"], "requires": f"{CORE_DISTRIBUTION}=={version}",
            "arches": sorted(arches), "directory": row["directory"], "code_format": "native"}  # glue: canonicalize architecture identifiers in metadata
    if bundled_ptx is not None:
        if vendor != "cuda":
            raise ValueError("only the NVIDIA vendor package can bundle PTX")
        validate_bundle_descriptor(bundled_ptx)
        marker["bundled_ptx"] = dict(bundled_ptx)
    return marker


def payload_marker(profile, version, arches):
    row = PAYLOADS[profile]
    return {"schema": PAYLOAD_SCHEMA, "role": "payload", "vendor": row["vendor"],
            "version": version, "distribution": row["distribution"],
            "requires": f"{CORE_DISTRIBUTION}=={version}", "arches": sorted(arches),  # glue: canonicalize architecture identifier strings in package metadata
            "directory": row["directory"], "code_format": row["code_format"]}


def package_requirements(profile, version):
    if profile == CORE_PROFILE:
        return core_requirements(version)
    package(profile)  # Validate that the profile is registered.
    return [f"{CORE_DISTRIBUTION}=={version}"]


def package_marker(profile, version, arches=(), bundled_ptx=None):
    if bundled_ptx is not None and profile != "nvidia":
        raise ValueError("only the NVIDIA vendor package can bundle PTX")
    if profile == CORE_PROFILE:
        return core_marker(version)
    if package(profile)["role"] == "vendor":
        return plugin_marker(by_profile(profile), version, arches, bundled_ptx=bundled_ptx)
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


BUNDLED_PTX_ROOT = "mojolearn/cuda_ptx/sm_80"
BASELINE_ADMISSION = "PTX_IDENTITY_ADMISSION.json"


#: binding_origin labels of bundled PTX bytes in a release inventory. Only a
#: bundle that carries a release identity admission may be called qualified.
BUNDLED_PTX_ORIGIN_ADMITTED = "qualified-ptx-bundle"
BUNDLED_PTX_ORIGIN_FALLBACK = "ptx-fallback-bundle"


def validate_bundle_descriptor(descriptor):
    """The vendor marker binds the PTX payload manifest, and the separate
    release identity admission when the wheel ships one. A manifest-only
    descriptor is the FAST/DETERMINISTIC fallback with no IDENTICAL claim."""
    import re
    if (not isinstance(descriptor, dict)
            or set(descriptor) not in ({"manifest_sha256"}, {"manifest_sha256", "admission_sha256"})
            or not all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)
                       for value in descriptor.values())):  # glue: check one or two SHA256 metadata fields
        raise ValueError("invalid bundled PTX digest descriptor")


def bundled_ptx_origin(descriptor):
    """The inventory origin label a bundle descriptor earns."""
    validate_bundle_descriptor(descriptor)
    return BUNDLED_PTX_ORIGIN_ADMITTED if "admission_sha256" in descriptor else BUNDLED_PTX_ORIGIN_FALLBACK


def owns_member(profile, member, bundled_ptx=None):
    """Ownership needs package context: experimental and bundled PTX collide."""
    owner = member_payload(member)
    if owner == "nvidia-ptx80" and profile == "nvidia" and bundled_ptx is not None:
        validate_bundle_descriptor(bundled_ptx)
        return member.startswith(BUNDLED_PTX_ROOT + "/")
    return owner == profile


def validate_bundled_ptx(descriptor, manifest_bytes, admission_bytes, files):
    """Validate transported bundle bytes; return the separate admission record,
    or None for a manifest-only bundle (which must carry no admission bytes).

    files is the complete relative GPU .so SHA256 map under cuda_ptx/sm_80.
    No record is generated and no measured configuration is broadened here.
    """
    import hashlib
    import importlib.util
    import json
    from pathlib import Path
    validate_bundle_descriptor(descriptor)
    admitted = "admission_sha256" in descriptor
    if admitted != (admission_bytes is not None):
        raise ValueError("bundled PTX admission bytes and vendor marker disagree")
    if (hashlib.sha256(manifest_bytes).hexdigest() != descriptor["manifest_sha256"]
            or (admitted and hashlib.sha256(admission_bytes).hexdigest() != descriptor["admission_sha256"])):
        raise ValueError("bundled PTX metadata bytes differ from vendor marker")
    manifest = json.loads(manifest_bytes)
    validate_baseline_manifest(manifest, files)
    if manifest.get("source_dirty") is not False:
        raise ValueError("bundled PTX needs a clean source manifest")
    if not admitted:
        return None
    admission = json.loads(admission_bytes)
    # This module is also loaded without package initialization by build tools.
    spec = importlib.util.spec_from_file_location("_mojolearn_ptx_admission", Path(__file__).with_name("ptx_admission.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.validate_admission(admission, source_commit=manifest["source_commit"],
                              manifest_sha256=descriptor["manifest_sha256"])
    return admission
