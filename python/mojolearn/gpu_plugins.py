# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINUX GPU PLUGIN PACKAGES: one table, read by the loader and the packer.

On Linux mojolearn ships as three PyPI projects (2026-09-25), the plugin
pattern JAX and CuPy use, so NVIDIA and AMD can release independently:

    mojolearn        pure Python, the host (CPU) bindings under mojolearn/host/
                     and the MAX runtime closure under mojolearn/.libs/
    mojolearn-cuda   ONLY mojolearn/cuda/<arch>/... (every tier of every
                     NVIDIA architecture the release carries)
    mojolearn-rocm   ONLY mojolearn/hip/<arch>/...  (AMD, gfx942)

`pip install "mojolearn[cuda]"` pulls the NVIDIA plugin, `[rocm]` the AMD
one. The versions are locked both ways: the core's extra pins the plugin at
exactly its own version and the plugin requires exactly its core.

A PLUGIN INSTALLS INTO THE CORE'S OWN PACKAGE DIRECTORY, at the very paths
the single combined wheel used (mojolearn/cuda/sm_90a/identical/...). That is
deliberate and it is the identity argument: every binding carries the RUNPATH
`$ORIGIN/../../.libs` family that build_sets.sh gave it on the box, which
resolves to mojolearn/.libs only from there, so the set bytes, their paths and
the way the dynamic linker resolves them are exactly those of the combined
wheel. Nothing is relinked, renamed or preloaded. The plugin owns no Python
module and no __init__.py, so it never overwrites a file of the core.

The macOS wheel is unchanged: its Metal sets stay inside `mojolearn`.

This module imports nothing from the package, so the packer and the tools
load it by path (the host_surface.py pattern).
"""

#: vendor directory -> the plugin that ships it. `vendor` is the directory
#: name under mojolearn/ and the string `<prefix>_vendor()` answers.
PLUGINS = {
    "cuda": {
        "distribution": "mojolearn-cuda",
        "wheel_name": "mojolearn_cuda",
        "extra": "cuda",
        "profile": "cuda",
        "label": "NVIDIA (CUDA)",
    },
    "hip": {
        "distribution": "mojolearn-rocm",
        "wheel_name": "mojolearn_rocm",
        "extra": "rocm",
        "profile": "rocm",
        "label": "AMD (ROCm/HIP)",
    },
}

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
CORE_SCHEMA = "mojolearn.gpu-plugins.v1"
PLUGIN_SCHEMA = "mojolearn.gpu-plugin.v1"


def vendors():
    """The vendor directories a plugin can ship, in table order."""
    return tuple(PLUGINS)


def plugin(vendor):
    """The table row of `vendor` ('cuda' or 'hip')."""
    return PLUGINS[vendor]


def by_profile(profile):
    """vendor for a plugin profile name ('cuda' -> 'cuda', 'rocm' -> 'hip')."""
    for vendor, row in PLUGINS.items():
        if row["profile"] == profile:
            return vendor
    raise KeyError(profile)


def member_vendor(arcname):
    """Which plugin owns a wheel member: the vendor for
    `mojolearn/<vendor>/...`, None for everything the core ships."""
    parts = arcname.split("/")
    if len(parts) > 2 and parts[0] == "mojolearn" and parts[1] in PLUGINS:
        return parts[1]
    return None


def install_command(vendor, version=None):
    """The pip command that installs the plugin for `vendor`."""
    pin = f"=={version}" if version else ""
    return f'pip install "mojolearn[{PLUGINS[vendor]["extra"]}]{pin}"'


def core_marker(version):
    """The core's marker document."""
    return {"schema": CORE_SCHEMA, "version": version,
            "plugins": {v: {"distribution": r["distribution"], "extra": r["extra"]}
                        for v, r in PLUGINS.items()}}


def plugin_marker(vendor, version, arches):
    """A plugin's marker document."""
    return {"schema": PLUGIN_SCHEMA, "vendor": vendor, "version": version,
            "distribution": PLUGINS[vendor]["distribution"],
            "requires": f"{CORE_DISTRIBUTION}=={version}", "arches": sorted(arches)}
