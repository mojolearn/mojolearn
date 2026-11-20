# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN body of `bindings/manifold_errors_native.mojo` (wave 8 run split): the
typed text refinement of a manifold kernel error, moved unchanged. The door
keeps the CPython exception fetch, restore and cause chaining."""


def _refinement(text: String, kind: Int, rows: Int, cols: Int, fitting: Bool,
                 name: String) -> Optional[String]:
    for marker in ["device_context", "CUDA", "HIP", "hipError", "Metal", "MTL"]:  # small-loop(device fault markers: existing public diagnostic exclusions)
        if marker in text:
            return Optional[String]()
    if kind == 0:
        for marker in ["NaN", "Infinit", "infinit", "nonfinite", "non-finite", "not finite", "requires finite", "must be finite"]:  # small-loop(nonfinite markers: existing public diagnostic spellings)
            if marker in text:
                var what = String("Input") if fitting else String("Input X")
                return Optional(what + " contains NaN or infinity (" + name + ": " + text + ")")
    if fitting and rows == 1 and kind <= 2:
        return Optional(name + " cannot be fit on n_samples=1: " + text)
    if fitting and cols == 1 and kind == 1:
        return Optional(name + " cannot be fit on n_features=1: " + text)
    return Optional[String]()
