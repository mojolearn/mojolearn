# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The float32 word pointer shared by the input casts (core/cast_in.mojo,
core/label_cast.mojo) and the sequence ops (sequence/ops.mojo, which defined
it until lane thin-edges-1, 2026-11-28, and still exports it). It lives here
so that a family that only casts its input does not import the sequence ops."""
comptime FP = MutPointer[Float32, MutUntrackedOrigin]
