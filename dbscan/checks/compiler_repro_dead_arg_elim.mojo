# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""COMPILER REPRO, not a check. `DeadArgumentElimination surveyUse failed`.

    Mojo 1.0.0 (ed45d567), the pinned toolchain in pixi.lock.
    Reduced 2026-09-09 on Linux x86-64 (NVIDIA L40S RunPod pod, driver
    595.91.07); first seen 2026-09-01 on an Apple M4 (macOS, arm64).

    mojo build -O3 -I . dbscan/checks/compiler_repro_dead_arg_elim.mojo
        DeadArgumentElimination surveyUse failed.
        UNREACHABLE executed!
        (abort, exit 134)
    mojo build -O2 ...   asserts the same way
    mojo build -O1 ...   builds and prints "1.0000151 1.000015"
    mojo build -O0 ...   builds

THE CONSTRUCT. A `def` that takes a `List` by reference and an `Int`, whose
`while` condition re-reads `len(list)` every iteration AND whose induction
variable steps by the runtime `Int` argument. Both halves are needed:

    while k < len(w): ...; k += width        ASSERTS      (this file)
    var n = len(w); while k < n: k += width  builds       (hoisted bound)
    for k in range(0, len(w), width): ...    builds       (strided range)
    while k < len(w): ...; k += 1            builds       (unit stride)
    while k < len(w): ...; k += Int(w[0])    builds       (stride not an arg)
    def f[width: Int](w): while k < len(w)   builds       (comptime stride)

Also measured on the same box: one call site or two, `@no_inline`, an owned
(`var`) list argument, a runtime-derived width, an inner `List` of partials
or none, `List[Int]` or `List[Float32]`, a second list indexed through the
first -- none of them change the outcome. `@always_inline` on the callee
builds, which is consistent with an argument-survey pass being the one that
trips.

This is `dbscan/checks/dbscan_check.mojo`'s `_host_weighted_degree_strided`
with everything else removed; that function now hoists the bound.
"""


def _strided(w: List[Float32], width: Int) -> Float32:
    var acc = Float32(0.0)
    var k = 0
    while k < len(w):
        acc = acc + w[k]
        k += width
    return acc


def main() raises:
    var vals = List[Float32]()
    var tiny = Float32(1.0) / Float32(16777216.0)
    for k in range(256):
        vals.append(Float32(1.0) if k == 0 else tiny)
    var wide = _strided(vals, 128)
    var narrow = _strided(vals, 64)
    if wide == narrow:
        raise Error("same: " + String(wide))
    print(String(wide) + " " + String(narrow))
