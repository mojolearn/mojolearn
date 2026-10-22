# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Int32 pointer alias the x_prep binding's fold entries read through.
TargetEncoder's cross-fit folds are built on the device on every tier
(`_expansion_prep._native_folds`); the host fold walks that lived here are
gone."""

comptime I32P = MutPointer[Int32, MutAnyOrigin]
