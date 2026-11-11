# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Compile-time accelerator predicates for the selected Mojolearn build target.

Mojo 1.2 replaced the vendor ``has_*_gpu_accelerator`` functions with
``default_accelerator().is_*_gpu()``. The target query requires an accelerator,
so preserve the original false result on accelerator-free CPU builds before
instantiating it. These helpers are evaluated only by the comptime constants;
no runtime device enumeration, target override or routing policy is added.

https://mojolang.org/releases/v1.2.0/#removed
https://mojolang.org/docs/std/sys/info/default_accelerator/
"""

from std.sys import default_accelerator, has_accelerator


def _apple_target() -> Bool:
    comptime if has_accelerator():
        return default_accelerator().is_apple_gpu()
    return False


def _amd_target() -> Bool:
    comptime if has_accelerator():
        return default_accelerator().is_amd_gpu()
    return False


def _nvidia_target() -> Bool:
    comptime if has_accelerator():
        return default_accelerator().is_nvidia_gpu()
    return False


comptime TARGET_HAS_APPLE_GPU = _apple_target()
comptime TARGET_HAS_AMD_GPU = _amd_target()
comptime TARGET_HAS_NVIDIA_GPU = _nvidia_target()
