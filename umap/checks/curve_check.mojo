# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from umap.curve import fit_umap_curve


def main() raises:
    var default_curve = fit_umap_curve(Float32(0.1), Float32(1.0))
    if not (default_curve.a > Float32(0.0)) or not (
        default_curve.b > Float32(0.0)
    ):
        raise Error("UMAP default curve fit is not positive")
    var a_error = default_curve.a - Float32(1.57694346)
    if a_error < Float32(0.0):
        a_error = -a_error
    var b_error = default_curve.b - Float32(0.89506088)
    if b_error < Float32(0.0):
        b_error = -b_error
    if a_error > Float32(2.0e-4) or b_error > Float32(2.0e-4):
        raise Error("UMAP default curve disagrees with scipy reference")
    var custom = fit_umap_curve(Float32(0.35), Float32(1.7))
    if not (custom.a > Float32(0.0)) or not (custom.b > Float32(0.0)):
        raise Error("UMAP custom curve fit is not positive")
    if custom.a == default_curve.a and custom.b == default_curve.b:
        raise Error("UMAP curve fit ignored min_dist/spread")
    print("UMAP curve parameter fit PASS")
