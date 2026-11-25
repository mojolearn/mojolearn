# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UNPACK: a resolved `numeric_profile` (the canonical form the profile door
returns: a preset string, or an exact dict `{"preset": name, part: name,
...}`) into the native parts mask of `core/numeric_profile_parts.mojo`.

Called ONCE per model at construction or load (CausalLM, ParallelCausalLM,
TransformerBlock); the mask is what the native plan keeps. Validation of the
spelling (unknown parts, a format a part cannot use, a family without the
part) is the profile door's (`numeric_profile_native_door.mojo` `resolve`);
this reader still refuses anything it cannot place, by name, rather than
guess."""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_item
from core.numeric_profile_parts import (
    NP_PART_CONTEXT, np_mask_check, np_part_index, np_preset_mask,
)


def profile_parts_mask(spec: PythonObject) raises -> Int:
    """The parts mask of a resolved profile (module docstring)."""
    var b = native_loaded_module("builtins")
    if py_type(spec) is b.str:
        return np_preset_mask(String(py=spec))
    if py_type(spec) is not b.dict:
        raise Error("TypeError: mojolearn: numeric_profile must be a profile name or a dict "
                    + "{'preset': name, part: name, ...}")
    var preset = py_item(spec, PythonObject("preset"))
    if py_type(preset) is not b.str:
        raise Error("ValueError: mojolearn: a numeric_profile dict needs 'preset', a profile name")
    var mask = np_preset_mask(String(py=preset))
    var keys = py_call(b.list, py_call(spec.keys))
    for i in range(len(keys)):  # driver-loop: over the few part overrides of one model
        var key = keys[i]
        if py_type(key) is not b.str:
            raise Error("TypeError: mojolearn: numeric_profile part names are strings")
        var name = String(py=key)
        if name == "preset":
            continue
        var part = np_part_index(name)
        if part < 0:
            raise Error("ValueError: mojolearn: numeric_profile has no part " + name)
        var value = py_item(spec, key)
        if py_type(value) is not b.str:
            raise Error("TypeError: mojolearn: numeric_profile part " + name + " takes a profile name")
        var fmt = String(py=value)
        if fmt == "fixed15_v1":
            if part == NP_PART_CONTEXT:
                raise Error("ValueError: mojolearn: numeric_profile part attn_context cannot use fixed15_v1 "
                            + "(P.V's per-row scales would depend on the key span); it is fp32_v1 only")
            mask = mask | (1 << part)
        elif fmt == "fp32_v1":
            mask = mask & ~(1 << part)
        else:
            raise Error("ValueError: mojolearn: numeric_profile part " + name + " takes 'fixed15_v1' or "
                        + "'fp32_v1', got " + fmt)
    np_mask_check(mask)
    return mask
